from data.dataEmbedding import Embedding


def imgInputEmb(img_path):
     emb = Embedding(image_path=img_path)
     imgInputVec = emb.Image_embedding
     return imgInputVec


if __name__ == "__main__":
     path = "D:/study/OJT/project/image.png"
     print(imgInputEmb(path))