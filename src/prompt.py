import math
import re
import torch
from compel import ReturnedEmbeddingsType

class PromptManager: 

    def __init__(self, compel) -> None:
        self.nat_weights = re.compile("\((.+?):([\d\.]+)\)", re.MULTILINE | re.IGNORECASE)
        self.compel = compel


    # Compel creation
    # 2.0.2 - fix for pipeline.enable_sequential_cpu_offloading() with SDXL models (you need to pass device='cuda' on compel init)
    def createCompel(self, pipeline, prompt, negative_prompt, requires_pooled=[False, True]):
        in_compel = self.compel(
            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] ,
            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=requires_pooled
        )

        conditioning, pooled = in_compel(prompt)
        neg_conditioning, neg_pooled = in_compel(negative_prompt)

        return conditioning, pooled, neg_conditioning, neg_pooled
    

    def make_enclosures_symmetrical(self, prompt, open_char="(", close_char=")"):
        """Make the number of open and close characters in the prompt symmetrical.
        A prompt may have many sets of enclosed subprompts, and the number of open and close characters
        may be unbalanced. This function makes the number of open and close characters symmetrical by
        removing open or close characters to the beginning or end of the enclosed subprompts.
        """
        new_prompt = prompt
        subprompt_finder = re.compile(
            f"({re.escape(open_char)}+)([^{re.escape(close_char)}]+)({re.escape(close_char)}+)",
            re.MULTILINE | re.IGNORECASE,
        )
        subprompts = subprompt_finder.findall(prompt)
        for open_chars, subprompt, close_chars in subprompts:
            if len(open_chars) == len(close_chars):
                continue

            if len(open_chars) > len(close_chars):
                new_subprompt = (open_char * len(close_chars)) + subprompt + close_chars
            else:
                new_subprompt = open_chars + subprompt + (close_char * len(open_chars))
            new_prompt = new_prompt.replace(
                open_chars + subprompt + close_chars, new_subprompt
            )
        return new_prompt


    # Copied without modification from nataili hopefully in accordance with their licensing terms. Thanks nataili!
    def rewrite_a1111_style_weights(self, prompt):
        prompt = self.make_enclosures_symmetrical(prompt, "(", ")")
        prompt = self.make_enclosures_symmetrical(prompt, "[", "]")

        def rewrite_for_char(
            prompt, open_char="(", close_char=")", max_chars=5, weight_basis=1.1
        ):
            # Iterate over the maximum number of modifier characters downwards
            for num_chars in range(max_chars, 0, -1):
                open = open_char * num_chars
                close = close_char * num_chars

                # Find subprompt with num_chars chars
                subprompt_open_i = prompt.find(open)
                subprompt_close_i = prompt.find(close, subprompt_open_i + 1)
                while subprompt_open_i != -1 and subprompt_close_i != -1:
                    subprompt = prompt[subprompt_open_i + num_chars : subprompt_close_i]
                    og_subprompt = subprompt
                    weight = None

                    # if subprompt contains a ":" use that weight as the base weight
                    if subprompt.find(":") != -1 and num_chars > 1:
                        subprompt, weight = subprompt.split(":")
                        if not weight:
                            weight = 1 / weight_basis
                        else:
                            weight = float(weight) * math.pow(weight_basis, num_chars)

                    # otherwise use the weight basis
                    elif subprompt.find(":") == -1:
                        weight = math.pow(weight_basis, num_chars)

                    elif subprompt.find(":") != -1 and num_chars == 1:
                        subprompt, weight = subprompt.split(":")
                        if not weight:
                            weight = 1
                        else:
                            weight = float(weight)

                    # Replace the prompt with the nataili-style prompt
                    if weight is not None:
                        prompt = prompt.replace(
                            open + og_subprompt + close, f"({subprompt}:{weight:.2f})"
                        )

                    # Find next subprompt
                    subprompt_open_i = prompt.find(open, subprompt_open_i + 1)
                    subprompt_close_i = prompt.find(close, subprompt_open_i + 1)
            return prompt

        # Rewrite for ( and ) trains
        prompt = rewrite_for_char(
            prompt, open_char="(", close_char=")", max_chars=5, weight_basis=1.1
        )
        # Rewrite for [ and ] trains
        prompt = rewrite_for_char(
            prompt, open_char="[", close_char="]", max_chars=5, weight_basis=0.9
        )

        return prompt


    def rewrite_prompt_for_compel(self, prompt):
        prompt = self.rewrite_a1111_style_weights(prompt)

        # convert the prompt weighting syntax from `a tall (man:1.1) picking apricots` to `a tall (man)1.1 picking apricots`
        prompt = self.nat_weights.sub(r"(\1)\2", prompt)

        return prompt

    # not used
    def get_prompt_embeds(self, prompt):
        prompt = self.rewrite_prompt_for_compel(prompt)
        with torch.no_grad():
            conditioning = self.compel.build_conditioning_tensor(prompt)
        return conditioning


    # not used
    def getCompelValue(self, prompt, negative_prompt: None): 
        prompt_embeds = self.get_prompt_embeds(prompt)

        if negative_prompt is not None:
            negative_prompt_embeds = self.get_prompt_embeds(negative_prompt)
        else:
            negative_prompt_embeds = self.get_prompt_embeds("")

        [
            prompt_embeds,
            negative_prompt_embeds
        ] = self.compel.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])

        return [ prompt_embeds, negative_prompt_embeds ]
