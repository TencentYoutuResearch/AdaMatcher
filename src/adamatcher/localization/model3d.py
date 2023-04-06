import logging
import os
import pdb
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc
from tqdm import tqdm

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
from src.adamatcher.utils.colmap import read_model
from src.adamatcher.utils.io import parse_image_lists, parse_retrieval
from src.adamatcher.utils.quaternions import weighted_pose
from src.adamatcher.utils.wrappers import Camera, Pose

# from ..utils import read_model
# from ..utils import weighted_pose
# from ..utils import parse_image_lists, parse_retrieval
# from ..utils import Camera, Pose

logger = logging.getLogger(__name__)


class Model3D:
    def __init__(self, path):
        logger.info('Reading COLMAP model %s.', path)
        self.cameras, self.dbs, self.points3D = read_model(path)
        self.name2id = {i.name: i.id for i in self.dbs.values()}

    def covisbility_filtering(self, dbids):
        clusters = do_covisibility_clustering(dbids, self.dbs, self.points3D)
        dbids = clusters[0]
        return dbids

    def pose_approximation(self, qname, dbids, global_descriptors, alpha=8):
        """Described in:

        Benchmarking Image Retrieval for Visual Localization. Noé Pion, Martin
        Humenberger, Gabriela Csurka, Yohann Cabon, Torsten Sattler. 3DV 2020.
        """
        dbs = [self.dbs[i] for i in dbids]

        dbdescs = np.stack([global_descriptors[im.name] for im in dbs])
        qdesc = global_descriptors[qname]
        sim = dbdescs @ qdesc
        weights = sim**alpha
        weights /= weights.sum()

        tvecs = [im.tvec for im in dbs]
        qvecs = [im.qvec for im in dbs]
        return weighted_pose(tvecs, qvecs, weights)

    def get_dbid_to_p3dids(self, p3did_to_dbids):
        """Link the database images to selected 3D points."""
        dbid_to_p3dids = defaultdict(list)
        for p3id, obs_dbids in p3did_to_dbids.items():
            for obs_dbid in obs_dbids:
                dbid_to_p3dids[obs_dbid].append(p3id)
        return dict(dbid_to_p3dids)

    def get_p3did_to_dbids(
        self,
        dbids: List,
        loc: Optional[Dict] = None,
        inliers: Optional[List] = None,
        point_selection: str = 'all',
        min_track_length: int = 3,
    ):
        """Return a dictionary mapping 3D point ids to their covisible dbids.

        This function can use hloc sfm logs to only select inliers. Which can
        be further used to select top reference images / in sufficient track
        length selection of points.
        """
        p3did_to_dbids = defaultdict(set)
        if point_selection == 'all':
            for dbid in dbids:
                p3dids = self.dbs[dbid].point3D_ids
                for p3did in p3dids[p3dids != -1]:
                    p3did_to_dbids[p3did].add(dbid)
        elif point_selection in ['inliers', 'matched']:
            if loc is None:
                raise ValueError('"{point_selection}" point selection requires'
                                 ' localization logs.')

            # The given SfM model must match the localization SfM model!
            for (p3did, dbidxs), inlier in zip(loc['keypoint_index_to_db'][1],
                                               inliers):
                if inlier or point_selection == 'matched':
                    obs_dbids = set(loc['db'][dbidx] for dbidx in dbidxs)
                    obs_dbids &= set(dbids)
                    if len(obs_dbids) > 0:
                        p3did_to_dbids[p3did] |= obs_dbids
        else:
            raise ValueError(f'{point_selection} point selection not defined.')

        # Filter unstable points (min track length)
        p3did_to_dbids = {
            i: v
            for i, v in p3did_to_dbids.items()
            if len(self.points3D[i].image_ids) >= min_track_length
        }

        return p3did_to_dbids

    def rerank_and_filter_db_images(self,
                                    dbids: List,
                                    ninl_dbs: List,
                                    num_dbs: int,
                                    min_matches_db: int = 0):
        """Re-rank the images by inlier count and filter invalid images."""
        dbids = [
            dbids[i] for i in np.argsort(-ninl_dbs)
            if ninl_dbs[i] > min_matches_db
        ]
        # Keep top num_images matched image images
        dbids = dbids[:num_dbs]
        return dbids

    def get_db_inliers(self, loc: Dict, dbids: List, inliers: List):
        """Get the number of inliers for each db."""
        inliers = loc['PnP_ret']['inliers']
        dbids = loc['db']
        ninl_dbs = np.zeros(len(dbids))
        for (_, dbidxs), inl in zip(loc['keypoint_index_to_db'][1], inliers):
            if not inl:
                continue
            for dbidx in dbidxs:
                ninl_dbs[dbidx] += 1
        return ninl_dbs


def do_covisibility_clustering(frame_ids, all_images, points3D):
    clusters = []
    visited = set()

    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = all_images[exploration_frame].point3D_ids
            connected_frames = set(j for i in observed if i != -1
                                   for j in points3D[i].image_ids)
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


# def run_query(model3d: Model3D, name: str, camera: Camera):
#     dbs = [model3d.name2id[r] for r in self.retrieval[name]]
#     loc = None  # if self.logs is None else self.logs[name]
#     ret = self.refiner.refine(name, camera, dbs, loc=loc)
#     return ret


def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    return image


base_default_config = dict(
    layer_indices=None,
    min_matches_db=10,
    num_dbs=1,
    min_track_length=3,
    min_points_opt=10,
    point_selection='all',
    average_observations=False,
    normalize_descriptors=True,
    compute_uncertainty=True,
)
conf = oc.create(base_default_config)

default_paths = dict(
    query_images=
    './datasets/ft_local/image-matching-toolbox-main/data/datasets/AachenDayNight/images_upright/',
    reference_images=
    './datasets/ft_local/image-matching-toolbox-main/data/datasets/AachenDayNight/images_upright/',
    reference_sfm=
    './datasets/ft_local/image-matching-toolbox-main/data/datasets/AachenDayNight/3D-models/aachen_v_1_1/',
    query_list=
    './datasets/ft_local/image-matching-toolbox-main/data/datasets/AachenDayNight/queries/*_time_queries_with_intrinsics.txt',
    global_descriptors='aachen_tf-netvlad.h5',
    retrieval_pairs='pairs-query-netvlad50.txt',
    results='pixloc_Aachen.txt',
)

if __name__ == '__main__':
    model3d_path = os.path.join(
        './datasets/ft_local/image-matching-toolbox-main/data/datasets/AachenDayNight/3D-models/aachen_v_1_1/'
    )
    query_txt_path = Path(
        os.path.join(
            './datasets/ft_local/image-matching-toolbox-main/data/datasets/AachenDayNight/queries/*_time_queries_with_intrinsics.txt'
        ))
    retrieval_pairs_path = Path(
        os.path.join(
            './datasets/ft_local/image-matching-toolbox-main/data/pairs/aachen_v1.1/pairs-query-netvlad50.txt'
        ))
    reference_images_path = Path(
        os.path.join(
            './datasets/ft_local/image-matching-toolbox-main/data/datasets/AachenDayNight/images_upright/'
        ))
    query_images_path = Path(
        os.path.join(
            './dataset/ft_local/image-matching-toolbox-main/data/datasets/AachenDayNight/images_upright/'
        ))

    # pdb.set_trace()
    model3d = Model3D(model3d_path)
    cameras = parse_image_lists(query_txt_path, with_intrinsics=True)
    queries = {n: c for n, c in cameras}
    query_names = list(queries.keys())  # [::skip or 1]
    retrieval = parse_retrieval(retrieval_pairs_path)

    # refiner = RetrievalRefiner(
    #         self.device, self.optimizer, self.model3d, self.extractor, paths,
    #         self.conf.refinement, global_descriptors=global_descriptors)

    # pdb.set_trace()
    for qname in tqdm(query_names):
        # pdb.set_trace()
        camera = Camera.from_colmap(queries[qname])
        try:
            # ret = self.run_query(name, camera)
            dbs = [model3d.name2id[r] for r in retrieval[qname]]
            # init
            id_init = dbs[0]
            image_init = model3d.dbs[id_init]
            Rt_init = (image_init.qvec2rotmat(), image_init.tvec)
            T_init = Pose.from_Rt(*Rt_init)
            loc, inliers = None, None

            p3did_to_dbids = model3d.get_p3did_to_dbids(
                dbs, loc, inliers, conf.point_selection, conf.min_track_length)

            dbid_to_p3dids = model3d.get_dbid_to_p3dids(p3did_to_dbids)
            rnames = [model3d.dbs[i].name for i in dbid_to_p3dids.keys()]
            images_ref = [
                read_image(reference_images_path / n) for n in rnames
            ]
            # reference
            dbid_p3did_to_feats = dict()
            for idx, dbid in enumerate(dbid_to_p3dids):
                p3dids = dbid_to_p3dids[dbid]
                i_ref = images_ref[idx]
                i_rname = rnames[idx]
            # ret = self.refine_query_pose(name, camera, T_init, p3did_to_dbids, [1])

            # query
            image_query = read_image(query_images_path / qname)

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logger.info('Out of memory')
                torch.cuda.empty_cache()
                ret = {'success': False}
            else:
                raise
