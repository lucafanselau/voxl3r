import torch
from extern.mast3r.mast3r.model import AsymmetricMASt3R
from extern.mast3r.dust3r.dust3r.utils.image import load_images


# Taken from `sparse_ga.py` inside of mast3r
@torch.no_grad()
def inference(model: AsymmetricMASt3R, img1, img2, device):
    shape1 = torch.from_numpy(img1["true_shape"]).to(device, non_blocking=True)
    shape2 = torch.from_numpy(img2["true_shape"]).to(device, non_blocking=True)
    img1 = img1["img"].to(device, non_blocking=True)
    img2 = img2["img"].to(device, non_blocking=True)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
        return dec1, dec2, res1, res2

    # decoder 1-2
    dec1, dec2, res1, res2 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)
    # decoder 2-1
    # res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

    return {
        # encoder output
        "feat1": feat1,
        "feat2": feat2,
        "pos1": pos1,
        "pos2": pos2,
        # decoder output
        "dec1": dec1,
        "dec2": dec2,
        # head output
        "res1": res1,
        "res2": res2,
    }


def get_default_device():
    """Pick GPU (cuda > mps) and else CPU"""
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name=None, device=get_default_device()):
    weights_path = "naver/" + model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model


def predict(model, folder):
    # this is also the default for the demo
    image_size = 512
    images = load_images(folder, image_size)

    # naive way to make pairs out of the image, by grouping (1, 2), (3, 4), ...
    pairs_in = [([images[i], images[i + 1]]) for i in range(0, len(images), 2)]

    predictions = inference(
        model, pairs_in[0][0], pairs_in[0][1], device=get_default_device()
    )

    return predictions


def main():

    args = {
        "local_network": False,
        "server_name": None,
        "image_size": 512,
        "server_port": None,
        "weights": None,
        "model_name": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "device": "mps",
        "tmp_dir": None,
        "silent": False,
        "share": False,
        "gradio_delete_cache": None,
    }

    model = load_model(args["model_name"])
    predictions = predict(model, "data/test_images")
    print(predictions)


if __name__ == "__main__":
    main()
