
class IDetectionService:

    def get_selected_model(self):
        pass

    def service_name(self):
        pass

    def model_name(self):
        pass

    def load_model(self,model=None):
        pass

    def detect_objects(self, frame,threshold= 0.5):
        pass

    def get_object_detection_models(self):
        pass
      