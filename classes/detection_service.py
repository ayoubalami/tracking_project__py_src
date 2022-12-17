
class IDetectionService:

    def get_selected_model(self):
        pass

    def service_name(self):
        pass

    def load_model(self,model=None):
        pass

    def detect_objects(self, frame,threshold:float,nms_threshold:float):
        pass

    def get_object_detection_models(self):
        pass

    def clean_memory(self):
        pass