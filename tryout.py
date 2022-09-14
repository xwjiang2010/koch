import prediction.app.worker.ray_processors as ray_proc

import ray

ray.init()
tuneExec = ray_proc.RayTuneExecutor.remote()