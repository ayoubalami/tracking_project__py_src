# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from classes.detection_services.detection_service import IDetectionService
# from classes.tracking_service.tracking_service import TrackingMethodEnum
from utils_lib.enums import TrackingMethodEnum

import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track, TrackState
from scipy.spatial.distance import cosine
import time
from csv import writer
import random
import copy

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3,trackingMethodEnum=TrackingMethodEnum.ANCHOR_BASED, active_tracking_evaluation=False):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
 
        self.trackingMethodEnum=trackingMethodEnum
        self.active_tracking_evaluation=  active_tracking_evaluation
        self.file_name_time=str(time.perf_counter())


    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, W,H, x_ratio,y_ratio):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        self.H, self.y_ratio,self.W, self.x_ratio=H, y_ratio,W, x_ratio

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
 
        # Update track set.
        # print("len(self.tracks)")
        # print(len(self.tracks))
        # print("len(self.tracks)")
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            # if  H-track.detection.to_tlbr()[3] <5:
            #     track.state = TrackState.Deleted

            # if track.detection.to_xyah()
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
            
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

          
    def object_similarity_min_dot(self,obj_1,obj_2):

        # X_threshold=300
        # if len(obj_1)>0 and len(obj_2)>0:
        #     if (abs((obj_1[0][0][1]-obj_2[0][0][1])/self.x_ratio)>X_threshold) :
        #         return 1
            
        obj_1=[v[1] for v in obj_1 ]
        obj_2=[v[1] for v in obj_2 ]
        
        obj_1_by_len=[[],[],[]]
        obj_2_by_len=[[],[],[]]

        for v in obj_1:
            if len(v)==128:
                obj_1_by_len[0].append(v)
            elif len(v)==256:
                obj_1_by_len[1].append(v)
            else :
                obj_1_by_len[2].append(v)

        for v in obj_2:
            if len(v)==128:
                obj_2_by_len[0].append(v)
            elif len(v)==256:
                obj_2_by_len[1].append(v)
            else :
                obj_2_by_len[2].append(v)

        min_dis=1

        for i in range(3):
            if len(obj_1_by_len[i])>0 and len(obj_2_by_len[i]):
                a = np.asarray(obj_1_by_len[i]) / np.linalg.norm(obj_1_by_len[i], axis=1, keepdims=True)
                b = np.asarray(obj_2_by_len[i]) / np.linalg.norm(obj_2_by_len[i], axis=1, keepdims=True)
                min_=np.min(1. - np.dot(a, b.T))
                if min_<min_dis:
                    min_dis=min_

        return min_dis

    def object_similarity_min_2(self,obj_1,obj_2):
        # similarity_array=[]
        # Y_threshold=400
        # if (abs((obj_1[0][0][1]-obj_2[0][0][1])/self.x_ratio)>X_threshold) or (abs((obj_1[0][0][2]-obj_2[0][0][2])/self.y_ratio)>Y_threshold):
        #     return 1

        X_threshold=300
        if len(obj_1)>0 and len(obj_2)>0:
            if (abs((obj_1[0][0][1]-obj_2[0][0][1])/self.x_ratio)>X_threshold) :
                return 1
            
        min_score=1
        for i,anchor_obj_1 in enumerate(obj_1):
            for j,anchor_obj_2 in enumerate(obj_2):
                if len(anchor_obj_1[1])==len(anchor_obj_2[1]):
                    current_similarity=cosine(anchor_obj_1[1],anchor_obj_2[1])
                    if current_similarity<min_score:
                        min_score=current_similarity
        return min_score

    # def cosine_vector(self, obj_1, obj_2):
    #     if len(obj_1[1])!=len(obj_2[1]):
    #         return 1
    #     return cosine(obj_1[1],obj_2[1])


    # def object_similarity_min_np(self,obj_1,obj_2):
    #     # similarity_array=[]
    #     return np.min( np.array(  min([[self.cosine_vector(anchor_1, anchor_2) for anchor_1 in obj_1]
    #         for anchor_2 in obj_2])))
            
            
        # for i,anchor_obj_1 in enumerate(obj_1):
        #     for j,anchor_obj_2 in enumerate(obj_2):
        #         if len(anchor_obj_1[1])==len(anchor_obj_2[1]):
        #             current_similarity=cosine(anchor_obj_1[1],anchor_obj_2[1])
        #             if current_similarity<min_score:
        #                 min_score=current_similarity
        # return min_score

    # def object_similarity_score(self,obj_1,obj_2,similarity_threshold=0.1):
    #     similarity_array=[]
    #     # nb_match_array=[]
    #     # nb_test=0
    #     score_match=0
    #     for i,anchor_obj_1 in enumerate(obj_1):
    #         for j,anchor_obj_2 in enumerate(obj_2):
    #             if len(anchor_obj_1[1])==len(anchor_obj_2[1]):
    #                 simil=cosine(anchor_obj_1[1],anchor_obj_2[1])
    #                 # similarity_array.append([len(anchor_obj_1[1]), simil])
    #                 # nb_test+=1
    #                 # print(""+str(simil)+" : "+str(i)+" "+str(j) )
    #                 if simil<similarity_threshold/100:
    #                     score_match+=500
    #                 elif simil<similarity_threshold/50:
    #                     score_match+=100
    #                 elif simil<similarity_threshold/20:
    #                     score_match+=50
    #                 elif simil<similarity_threshold/10:
    #                     score_match+=10
    #                 elif simil<similarity_threshold/5:
    #                     score_match+=5
    #                 elif simil<similarity_threshold/2:
    #                     score_match+=2
    #                 elif simil<similarity_threshold:
    #                     score_match+=1
    
    #     # print(score_match)
    #     nb_best_match_similarity=-score_match
    #     # min_similarity=1 if len(similarity_array)==0 else np.min(np.array(similarity_array))
    #     # print(nb_test)
    #     # print(nb_match/nb_test)

    #     return nb_best_match_similarity

    # def object_similarity_avg(self,obj_1,obj_2):
    #     similarity_array=[]
    #     for i,anchor_obj_1 in enumerate(obj_1):
    #         for j,anchor_obj_2 in enumerate(obj_2):
    #             if len(anchor_obj_1[1])==len(anchor_obj_2[1]):
    #                 current_sim=cosine(anchor_obj_1[1],anchor_obj_2[1])
    #                 if current_sim<.4:
    #                     similarity_array.append([len(anchor_obj_1[1]),current_sim])
    #     return 1 if len(similarity_array)==0 else np.mean(np.array(similarity_array))


    def object_association(self,detected_data, active_targets_data):
        cost_matrix=[]
        for i in range(len(active_targets_data)):
            cost_array=[]
            for j in range(len(detected_data)):
                cost_array.append(self.object_similarity_min_2(active_targets_data[i],detected_data[j]))
            cost_matrix.append(cost_array)
        return np.array(cost_matrix)

    
    def object_association_dot(self,detected_data, active_targets_data):
        
        cost_matrix=[]
        for i in range(len(active_targets_data)):
            cost_array=[]
            for j in range(len(detected_data)):
                cost_array.append(self.object_similarity_min_dot(active_targets_data[i],detected_data[j]))
            cost_matrix.append(cost_array)
        
        # print(np.array(cost_matrix))
        return np.array(cost_matrix)

    false_matching=0
    total_matching=0
    miss_tracking=0
    false_tracking=0
    


    def _match(self, detections):
    
        def evaluator():

            tracks_for_cnn=self.tracks.copy()
            tracks_for_optim=self.tracks.copy()
            
            confirmed_tracks = [
                i for i, t in enumerate(tracks_for_cnn) if t.is_confirmed()]
            unconfirmed_tracks = [
                i for i, t in enumerate(tracks_for_cnn) if not t.is_confirmed()]

            confirmed_tracks_optim = [
                i for i, t in enumerate(tracks_for_optim) if t.is_confirmed()]
            unconfirmed_tracks_optim = [
                i for i, t in enumerate(tracks_for_optim) if not t.is_confirmed()]

            # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # features = self.encoder(frame, bboxes)
            # optimized_gated_metric TO gated_metric
            # detections.
            
            #  rate of disabling detection to simulate detection loses
            eval_detections=copy.deepcopy(detections)

            #   gated_metric_for_sort OR optimized_gated_metric
            evaluated_func=None
            if self.trackingMethodEnum==TrackingMethodEnum.DEEP_SORT:
                evaluated_func=gated_metric
            if self.trackingMethodEnum==TrackingMethodEnum.SORT:
                evaluated_func=gated_metric_for_sort
            if self.trackingMethodEnum==TrackingMethodEnum.ANCHOR_BASED:
                evaluated_func=optimized_gated_metric

            matches_a_optim, unmatched_tracks_a_optim, unmatched_detections_optim = \
            linear_assignment.matching_cascade(
                evaluated_func, self.metric.matching_threshold, self.max_age,
                tracks_for_cnn, eval_detections, confirmed_tracks_optim)

            iou_track_candidates_optim = unconfirmed_tracks_optim + [
                k for k in unmatched_tracks_a_optim if
                tracks_for_optim[k].time_since_update == 1]

            unmatched_tracks_a_optim = [
                k for k in unmatched_tracks_a_optim if
                tracks_for_optim[k].time_since_update != 1]

            matches_b_optim, unmatched_tracks_b_optim, unmatched_detections_optim = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, tracks_for_optim,
                    eval_detections, iou_track_candidates_optim, unmatched_detections_optim)
                    
            matches_optim = matches_a_optim + matches_b_optim

            unmatched_tracks_optim = list(set(unmatched_tracks_a_optim + unmatched_tracks_b_optim))

            ##########""

            matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                tracks_for_cnn, detections, confirmed_tracks)

            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if
                tracks_for_cnn[k].time_since_update == 1]

            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if
                tracks_for_cnn[k].time_since_update != 1]

            matches_b, unmatched_tracks_b, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, tracks_for_cnn,
                    detections, iou_track_candidates, unmatched_detections)
                    
            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
 
            self.total_matching+=max(len(matches_optim),len(matches))
            self.miss_tracking+=len(set(matches)-set(matches_optim))
            self.false_tracking+=len(set(matches_optim)-set(matches))
            # print(f" False trakckng {self.false_tracking}")
            
            if (len(set(matches)-set(matches_optim))):
                print(matches)
                print(matches_optim)

            if (len(set(matches_optim)-set(matches))):
                print(matches)
                print(matches_optim)

            print(f" miss_tracking {self.miss_tracking}/{ self.total_matching}")
            print(f" false_tracking {self.false_tracking}/{ self.total_matching}")
            print("====>")
      

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            # print("STEP 1 DeepFeature =>>")
            # print(cost_matrix)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix

        def gated_metric_for_sort(tracks, dets, track_indices, detection_indices):
            features = np.array([[] for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # cost_matrix = self.metric.distance(features, targets)
            cost_matrix = np.ones((len(targets), len(features)))
            # print("STEP 1 DeepFeature =>>")
            # print(cost_matrix)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix

        def optimized_gated_metric(tracks, dets, track_indices, detection_indices):
            detected_data = [dets[i].object_data for i in detection_indices]
            active_targets_data = [tracks[i].detection.object_data for i in track_indices]
            
            # start_t=time.process_time()
            cost_matrix=self.object_association_dot(detected_data, active_targets_data)
            # cost_matrix = np.random.random((len(active_targets_data), len(detected_data)))

            # print(f"object_association_dot : {time.process_time()-start_t}")
            # print(cost_matrix)
            # start_t=time.process_time()
            # cost_matrix=self.object_association(detected_data, active_targets_data)
            # print(f"object_association : {time.process_time()-start_t}")
            # print(cost_matrix)

            # print("STEP 1 Optimized Association=>>")
            # print(cost_matrix)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix
        # Split track set into confirmed and unconfirmed tracks.

        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        if self.active_tracking_evaluation:
            evaluator()
            matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        else:
            # Associate confirmed tracks using appearance features.
            if self.trackingMethodEnum==TrackingMethodEnum.DEEP_SORT:
                matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks)
            elif self.trackingMethodEnum==TrackingMethodEnum.SORT:
                matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric_for_sort, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks)
            elif self.trackingMethodEnum==TrackingMethodEnum.ANCHOR_BASED:
                matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    optimized_gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks)
 
        #ASSOCIATION STRATEGY
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
                
        matches = matches_a + matches_b

        # if len(matches)>0:
        #     csv_file = "csv_results/matches"+ self.file_name_time+".csv"
        #     with open(csv_file, 'a') as csvfile:
        #         writer(csvfile).writerow(matches)

        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

            
    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_name = detection.get_class()
        track=Track(detection,
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, class_name)
        self.tracks.append(track)
        track.detection_bbox=detection
        self._next_id += 1
