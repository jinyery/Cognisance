import numpy as np
import os.path as path
from tempfile import mkdtemp
from sklearn.metrics import pairwise_distances as pair_dist

CUR_DIR = path.abspath(path.dirname(__file__))


class CoarseNode:
    # All ids are ids in forest.
    def __init__(self, agent, members, id=None, ldr_id=None, sub_ids=None):
        self.id = id
        self.ldr_id = ldr_id
        self.sub_ids = sub_ids

        self.agent = agent
        self.members = members

    def __len__(self):
        return len(self.members)

    def __str__(self):
        return "id: {}, ldr_id: {}, sub_ids: {}, agent: {}, members: {}".format(
            self.id, self.ldr_id, self.sub_ids, self.agent, self.members
        )

    def has(self, member: int) -> bool:
        return member in self.members

    def add_member(self, mem_id):
        if self.members is None:
            self.members = list()
        self.members.append(mem_id)

    def add_subordinate(self, sub_id):
        if self.sub_ids is None:
            self.sub_ids = list()
        self.sub_ids.append(sub_id)


class CoarseLeadingForest:
    def __init__(
        self,
        samples: list[list[int]],
        metric="euclidean",
        min_dist=None,
        max_dist=None,
        max_sample_size=10000,
    ) -> None:
        self.metric = metric
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_sample_size = max_sample_size

        self.root_ids: list[int] = list()
        self.coarse_nodes: list[CoarseNode] = list()
        self.compute_leader(np.array(samples, dtype=np.float32))

    def num_tree(self):
        return len(self.root_ids)

    def num_node(self):
        return len(self.coarse_nodes)

    def num_all_node(self):
        num_all_node = 0
        for node in self.coarse_nodes:
            num_all_node += len(node)
        return num_all_node

    # return coarse_node_id
    def where_is_fine_node(self, fine_node_id):
        for i, coarse_node in enumerate(self.coarse_nodes):
            if coarse_node.has(fine_node_id):
                return i
        return -1

    # return tree_id
    def where_is_coarse_node(self, coarse_node_id):
        tmp_id = coarse_node_id
        tmp_ldr_id = self.coarse_nodes[tmp_id].ldr_id
        while tmp_ldr_id != -1:
            tmp_id = tmp_ldr_id
            tmp_ldr_id = self.coarse_nodes[tmp_ldr_id].ldr_id
        return self.root_ids.index(tmp_id)

    def _compute_distance(self, samples: np.array) -> np.array:
        if len(samples) <= self.max_sample_size:
            dist = pair_dist(samples, metric=self.metric).astype(np.float32)
        else:
            tmp_file = path.join(mkdtemp(), "clf_dist.dat")
            dist = np.memmap(
                tmp_file, dtype="float32", mode="w+", shape=(len(samples), len(samples))
            )
            for i in range(0, len(samples), self.max_sample_size):
                end = i + self.max_sample_size
                if end > len(samples):
                    end = len(samples)
                dist[i:end] = pair_dist(samples[i:end], samples, metric=self.metric)

        if self.min_dist is None or self.max_dist is None:
            base = 0
            for i in range(len(samples)):
                top_k = np.sort(dist[i])[1:4]
                base += np.mean(top_k)
            base /= len(samples)
            self.min_dist = base * 0.3
            self.max_dist = base * 2.7
        return dist

    def _compute_density(self, dist: np.array) -> np.array:
        if len(dist) <= self.max_sample_size:
            tmp = np.exp(-((dist / self.max_dist) ** 2))
        else:
            tmp_file = path.join(mkdtemp(), "clf_dens.dat")
            tmp = np.memmap(
                tmp_file, dtype="float32", mode="w+", shape=(len(dist), len(dist))
            )
            for i in range(0, len(dist), self.max_sample_size):
                end = i + self.max_sample_size
                if end > len(dist):
                    end = len(dist)
                tmp[i:end] = np.exp(-((dist[i:end] / self.max_dist) ** 2))
        tmp[dist > self.max_dist] = 0
        np.fill_diagonal(tmp, 0)
        return np.sum(tmp, axis=0)

    def compute_leader(self, samples: np.array):
        self.root_ids.clear()
        self.coarse_nodes.clear()
        dist = self._compute_distance(samples)
        dens = self._compute_density(dist=dist)
        accessed = np.full(len(dens), False)
        density_argsort = np.argsort(dens)[::-1]
        for i in range(accessed.size):
            print_str = f"\033[7m\r[CLF:building node {i+1}/{accessed.size}]"
            print(print_str, end="")

            node = density_argsort[i]
            if accessed[node]:
                continue
            members = np.where(dist[node] <= self.min_dist)[0]
            not_accessed = np.where(accessed == False)[0]
            members = np.intersect1d(members, not_accessed)
            members = members.tolist()
            accessed[members] = True

            spare = np.where(dens >= dens[node])[0]
            abandoned = np.where(dist[node] > self.max_dist)[0]
            abandoned = np.append(abandoned, not_accessed)
            abandoned = np.append(abandoned, members)
            spare = np.setdiff1d(spare, abandoned)

            ldr_id = -1
            if spare.size > 0:
                tmp_idx = np.argmin(dist[node][spare])
                ldr_id = self.where_is_fine_node(spare[tmp_idx])
                assert ldr_id != -1
            node_id = len(self.coarse_nodes)
            coarse_node = CoarseNode(
                id=node_id, ldr_id=ldr_id, agent=node, members=members
            )
            if ldr_id == -1:
                self.root_ids.append(node_id)
            else:
                self.coarse_nodes[ldr_id].add_subordinate(node_id)
            self.coarse_nodes.append(coarse_node)

            print_end = "\033[0m\r" + " " * len(print_str) + "\r"
            print(print_end, end="")

    def leader_vector(self):
        leader_vector = np.full(len(self.coarse_nodes), -1)
        for i, node in enumerate(self.coarse_nodes):
            leader_vector[i] = node.ldr_id
        return leader_vector

    def generate_path(self, detailed=True):
        paths = list()
        detailed_paths = list()
        repetitions = np.zeros(self.num_node())
        detailed_repetitions = np.zeros(self.num_all_node())

        leader_vector = self.leader_vector()
        tmp1 = {i for i in range(len(leader_vector))}
        tmp2 = set(leader_vector.tolist())
        leaves_idx = tmp1 - tmp2
        for leaf_idx in leaves_idx:
            path = list()
            detailed_path = list()
            tmp_idx = leaf_idx
            while tmp_idx != -1:
                path.append(tmp_idx)
                repetitions[tmp_idx] += 1
                detailed_path.append(self.coarse_nodes[tmp_idx].members)
                detailed_repetitions[self.coarse_nodes[tmp_idx].members] += 1
                tmp_idx = int(leader_vector[tmp_idx])
            path.reverse()
            paths.append(path)
            detailed_path.reverse()
            detailed_paths.append(detailed_path)
        repetitions = repetitions.astype(int).tolist()
        detailed_repetitions = detailed_repetitions.astype(int).tolist()

        if not detailed:
            return paths, repetitions
        else:
            return detailed_paths, detailed_repetitions
