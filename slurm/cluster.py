import asyncio

import dask
from dask.utils import parse_timedelta
import dask_jobqueue
from distributed.deploy.cluster import Cluster


class PatchedSLURMCluster(dask_jobqueue.SLURMCluster):
    def __init__(self, *args, **kwargs):
        if "processes" not in kwargs:
            kwargs["processes"] = dask.config.get("jobqueue.slurm.processes")
        super().__init__(*args, **kwargs)

    def _update_worker_status(self, op, msg):
        if op == "remove":
            name = self.scheduler_info["workers"][msg]["name"]

            def f():
                mapping = {}
                siblings = {}
                for w in self.workers:
                    spec = self.worker_spec[w]
                    if "group" in spec:
                        siblings[w] = set()
                        for suffix in spec["group"]:
                            mapping[str(w) + suffix] = w
                            siblings[w].add(str(w) + suffix)
                    else:
                        mapping[w] = w
                        siblings[w] = {w}

                # print("mapping:", mapping)
                # print("siblings:", siblings)
                # print("scheduler:", self.scheduler_info["workers"])
                if (
                    name in mapping
                    and msg not in self.scheduler_info["workers"]
                    and not any(
                        d["name"] in siblings[mapping[name]]
                        for d in self.scheduler_info["workers"].values()
                    )
                ):
                    # print("remove worker")
                    worker = mapping[name]
                    self._futures.add(
                        asyncio.ensure_future(self.workers[worker].close())
                    )
                    del self.workers[worker]

            delay = parse_timedelta(
                dask.config.get("distributed.deploy.lost-worker-timeout")
            )

            asyncio.get_event_loop().call_later(delay, f)

        # call Cluster's method and not SpecCluster's method
        Cluster._update_worker_status(self, op, msg)
