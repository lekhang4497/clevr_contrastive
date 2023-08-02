from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from transformers import VisionTextDualEncoderModel
from transformers.models.clip.modeling_clip import (
    CLIPOutput,
    CLIPVisionConfig,
    CLIPVisionModel,
)


# shapes = ["cube", "sphere", "cylinder"]
# colors = ["gray", "red", "blue", "green", "yellow"]
# materials = ["rubber", "metal"]

# shape_captions = [f"The shape is {x}" for x in shapes]
# color_captions = [f"The color is {x}" for x in colors]
# material_captions = [f"The material is {x}" for x in materials]

# all_captions = shape_captions + color_captions + material_captions


# def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
#     return nn.functional.cross_entropy(
#         logits, torch.arange(len(logits), device=logits.device)
#     )


def multi_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


# def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
#     caption_loss = contrastive_loss(similarity)
#     image_loss = contrastive_loss(similarity.t())
#     return (caption_loss + image_loss) / 2.0


def prop_clip_loss(similarity: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    caption_loss = multi_bce_loss(similarity, labels)
    image_loss = multi_bce_loss(similarity.t(), labels.t())
    return (caption_loss + image_loss) / 2.0


class PropVisionTextModel(VisionTextDualEncoderModel):
    # def set_tokenized_props(self, tokenizerd_props: Dict[str, torch.LongTensor]):
    #     self.tokenized_props = tokenizerd_props

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CLIPOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        # if self.tokenized_props is None:
        #     raise ValueError("tokenized_props is None")

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # input_ids = self.tokenized_props["input_ids"]
        # attention_mask = self.tokenized_props["attention_mask"]

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]  # pooler_output
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]  # pooler_output
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = prop_clip_loss(logits_per_image, labels)

        if not return_dict:
            output = (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                text_outputs,
                vision_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
