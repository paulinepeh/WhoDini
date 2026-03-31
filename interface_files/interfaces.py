from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def detect(self, image):
        pass

class Embedder(ABC):
    @abstractmethod
    def get_embedding(self, image_crop):
        """Returns embedding vector or None"""
        pass

# from abc import ABC, abstractmethod

# class Detector(ABC):
#     @abstractmethod
#     def detect(self, image):
#         pass

# class Embedder(ABC):
#     @abstractmethod
#     def get_embedding(self, image_crop):
#         """Returns embedding vector or None"""
#         pass