# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from scipy.spatial.distance import cosine
import time


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

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3,use_cnn_feature_extraction=False):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.use_cnn_feature_extraction=use_cnn_feature_extraction
        
    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
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
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

        # self.lastActiveTrackes = {index: value for index, value in enumerate(active_targets)}
        # my_dict = {index: value for index, value in enumerate(active_targets)}


                # # max_w=min(int(anchor_obj_1[0][3]),int(anchor_obj_2[0][3]))*2
                # # max_h=min(int(anchor_obj_1[0][4]),int(anchor_obj_2[0][4]))*2
                # max_h=10
                # max_w=10
            
                # print(f"anchor_obj_1[0][1] = > {anchor_obj_1[0][1]} - {anchor_obj_2[0][1]}  = {abs(int(anchor_obj_2[0][1])-int(anchor_obj_1[0][1]))}")
                # print(f"anchor_obj_1[0][2] = > {anchor_obj_1[0][2]} - {anchor_obj_2[0][2]}  = {abs(int(anchor_obj_2[0][2])-int(anchor_obj_1[0][2]))}")
                  
                # if abs(int(anchor_obj_1[0][1])-int(anchor_obj_2[0][1]))>max_w or abs(int(anchor_obj_1[0][2])-int(anchor_obj_2[0][2]))>max_h:
                #     continue
                #     # print(max_w)
                #     # print(abs(int(anchor_obj_1[0][1])-int(anchor_obj_2[0][1])))
                     # similarity_array.append([len(anchor_obj_1[1]), cosine(anchor_obj_1[1],anchor_obj_2[1])])
                    # box1,box2=anchor_obj_1[0][1:5],anchor_obj_2[0][1:5]
                    # print(anchor_obj_1[0][0:5])
                    # print(anchor_obj_2[0][0:5])
                    # print(abs(box1[0]-box2[0]))
                    # print(abs(box1[1]-box2[1]))
                    # print(f"anchor_obj_1[0][1] = > {anchor_obj_1[0][1]}  {anchor_obj_2[0][1]} {abs(int(anchor_obj_2[0][1])-int(anchor_obj_1[0][1]))}")
                    # print(f"{anchor_obj_1[0][1]}  {anchor_obj_2[0][1]}")

    def object_similarity_avg_mins(self,obj_1,obj_2):
        similarity_array=[]
         # relative to 416
        for i,anchor_obj_1 in enumerate(obj_1):
            for j,anchor_obj_2 in enumerate(obj_2):
                if len(anchor_obj_1[1])==len(anchor_obj_2[1]):
                    similarity_array.append( cosine(anchor_obj_1[1],anchor_obj_2[1]) )
        if len(similarity_array)==0 :
            return 1
        
        # if len(similarity_array)>10:
        #     return np.mean(np.partition(np.array(similarity_array), 2)[:2])

        return np.min(np.array(similarity_array)) 

        # return 1 if len(similarity_array)==0 else np.mean(np.partition(similarity_array, 3)[:3])

    def object_similarity_min_2(self,obj_1,obj_2):
        # similarity_array=[]
        min_score=1
        for i,anchor_obj_1 in enumerate(obj_1):
            for j,anchor_obj_2 in enumerate(obj_2):
                if len(anchor_obj_1[1])==len(anchor_obj_2[1]):
                    current_similarity=cosine(anchor_obj_1[1],anchor_obj_2[1])
                    if current_similarity<min_score:
                        min_score=current_similarity
        return min_score


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
            # box1=active_targets_data[i][0]
            for j in range(len(detected_data)):
                # box2=detected_data[j][0]
                # start_time=time.perf_counter() 
                cost_array.append(self.object_similarity_min_2(active_targets_data[i],detected_data[j]))
                # cost_array.append(self.object_similarity_avg_mins(active_targets_data[i],detected_data[j]))
                # print(f" object_similarity_min : {time.perf_counter() -start_time}")
            cost_matrix.append(cost_array)

        
        # print(np.array(cost_matrix))
        return np.array(cost_matrix)
    
    false_matching=0
    total_matching=0
    def _match(self, detections):

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

        def optimized_gated_metric(tracks, dets, track_indices, detection_indices):
            detected_data = [dets[i].object_data for i in detection_indices]
            active_targets_data = [tracks[i].detection.object_data for i in track_indices]
            cost_matrix=self.object_association(detected_data, active_targets_data)
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

 
        # Associate confirmed tracks using appearance features.

        if self.use_cnn_feature_extraction:
            matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        else:
            matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                optimized_gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
                
# gated_metric
# optimized_gated_metric
        # print("<matches_a>")

        # print(matches_a)
        # print(matches_a_)
        # print(np.array(matches_a)==np.array(matches_a_))
        # print(np.array_equal(matches_a, matches_a_))

        # self.total_matching+=1
        # if not np.array_equal(matches_a, matches_a_):
        #     self.false_matching+=1

        # print(str(self.false_matching)+"/"+str(self.total_matching))
        # print("====>")
        

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
        # print(matches)


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
