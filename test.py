import ray
import os
import torch
from torch.distributed.device_mesh import init_device_mesh
from ray.dag.input_node import InputNode
from ray.dag.output_node import MultiOutputNode
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.single_controller.base.decorator import register, Dispatch
from transformers import AutoModelForCausalLM

class TestWorker(Worker):
    def __init__(self, role):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
       
        world_size = torch.distributed.get_world_size()
        print(f'world_size: {world_size}')
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=None,
            )
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def collect_weights(self, shape):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def send(self, shape):
        return torch.zeros(shape, device="cuda")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def recv(self, tensor):
        print(f"rollout worker {self.rank} received data: {tensor} on device {tensor.device}")
    
def create_worker_group(name, ray_cls, resource_pool_manager, num_workers=1):
    # Configure the resource pool
    ray_cls_with_init = RayClassWithInitArgs(ray.remote(ray_cls), role=name)
    worker_dict_cls = create_colocated_worker_cls(class_dict={name: ray_cls_with_init})
    worker_group = RayWorkerGroup(
        resource_pool=resource_pool_manager.get_resource_pool(name),
        ray_cls_with_init=worker_dict_cls,
    )
    return worker_group

ray.init()

num_send = 1 # number of TestSend workers
num_recv = 2 # number of TestRecv workers


# need to set up resource pool for correct communication setup via ray
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
resource_pool_spec = {
    "actor_pool_id": [num_send],
    "rollout_pool_id": [num_recv],
}
mapping = {
    "actor": "actor_pool_id",
    "rollout": "rollout_pool_id",
}
resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
resource_pool_manager.create_resource_pool()


send_workers = create_worker_group("actor", TestWorker, resource_pool_manager, num_send)
recv_workers = create_worker_group("rollout", TestWorker, resource_pool_manager, num_recv)

ray.get([getattr(worker, 'actor_collect_weights').remote() for worker in send_workers.workers])
# forward pass
with InputNode() as input_node:
    weights = getattr(send_workers.workers[0], "actor_send").bind(input_node).with_tensor_transport("nccl")        

    recv_workers_output = [
        getattr(worker, "rollout_recv").bind(weights)
        for worker in recv_workers.workers
    ]
    sync_param_dag = MultiOutputNode(recv_workers_output)

sync_param_dag = sync_param_dag.experimental_compile(_submit_timeout=10000)
ray.get(sync_param_dag.execute(1))

sync_param_dag.teardown()