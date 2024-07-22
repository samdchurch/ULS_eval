import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/app/MedSAM')
from segment_anything import sam_model_registry

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


if __name__ == '__main__':
    #MedSAM_CKPT_PATH = '/app/MedSAM/work_dir/MedSAM/lite_medsam.pth'
    MedSAM_CKPT_PATH = '/app/MedSAM/work_dir/MedSAM/medsam_vit_b.pth'

    if torch.cuda.is_available():
        print("CUDA is available! GPU can be used.")
        device = torch.device("cuda")
        print(f"Device name: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    