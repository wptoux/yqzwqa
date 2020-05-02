# -*- coding: utf-8 -*-
# +
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import joblib
from joblib import Parallel, delayed

from functools import partial
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
import numpy as np
from transformers import Pipeline, \
    PreTrainedTokenizer, PreTrainedModel, PreTrainedModel, ModelCard
from transformers.pipelines import ArgumentHandler
from squad import _is_whitespace, SquadFeatures, _new_check_is_max_context, \
    squad_convert_example_to_features, squad_convert_example_to_features_init, \
    remove_space_between_chinese, SquadFeatures, SquadExample


# -

class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped
    to internal SquadExample / SquadFeature structures.

    QuestionAnsweringArgumentHandler manages all the possible to create SquadExample from the command-line supplied
    arguments.
    """

    def __call__(self, *args, **kwargs):
        # Position args, handling is sensibly the same as X and data, so forwarding to avoid duplicating
        if args is not None and len(args) > 0:
            if len(args) == 1:
                kwargs["X"] = args[0]
            else:
                kwargs["X"] = list(args)

        # Generic compatibility with sklearn and Keras
        # Batched data
        if "X" in kwargs or "data" in kwargs:
            inputs = kwargs["X"] if "X" in kwargs else kwargs["data"]

            if isinstance(inputs, dict):
                inputs = [inputs]
            else:
                # Copy to avoid overriding arguments
                inputs = [i for i in inputs]

            for i, item in enumerate(inputs):
                if isinstance(item, dict):
                    if any(k not in item for k in ["question", "context"]):
                        raise KeyError("You need to provide a dictionary with keys {question:..., context:...}")

                    inputs[i] = QuestionAnsweringPipeline.create_sample(**item)

                elif not isinstance(item, SquadExample):
                    raise ValueError(
                        "{} argument needs to be of type (list[SquadExample | dict], SquadExample, dict)".format(
                            "X" if "X" in kwargs else "data"
                        )
                    )

            # Tabular input
        elif "question" in kwargs and "context" in kwargs:
            if isinstance(kwargs["question"], str):
                kwargs["question"] = [kwargs["question"]]

            if isinstance(kwargs["context"], str):
                kwargs["context"] = [kwargs["context"]]

            inputs = [
                QuestionAnsweringPipeline.create_sample(q, c) for q, c in zip(kwargs["question"], kwargs["context"])
            ]
        else:
            raise ValueError("Unknown arguments {}".format(kwargs))

        if not isinstance(inputs, list):
            inputs = [inputs]

        return inputs


# +
class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeline using ModelForQuestionAnswering head. See the
    `question answering usage <../usage.html#question-answering>`__ examples for more information.

    This question answering can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "question-answering", for answering questions given a context.

    The models that this pipeline can use are models that have been fine-tuned on a question answering task.
    See the list of available community models fine-tuned on such a task on
    `huggingface.co/models <https://huggingface.co/models?search=&filter=question-answering>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    default_input_names = "question,context"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        device: int = -1,
        task: str = "",
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=QuestionAnsweringArgumentHandler(),
            device=-1 if device == -1 else 0,
            task=task,
            **kwargs,
        )

    @staticmethod
    def create_sample(
        question: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Union[SquadExample, List[SquadExample]]:
        """
        QuestionAnsweringPipeline leverages the SquadExample/SquadFeatures internally.
        This helper method encapsulate all the logic for converting question(s) and context(s) to SquadExample(s).
        We currently support extractive question answering.
        Arguments:
             question: (str, List[str]) The question to be ask for the associated context
             context: (str, List[str]) The context in which we will look for the answer.

        Returns:
            SquadExample initialized with the corresponding question and context.
        """
        if isinstance(question, list):
            return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
        else:
            return SquadExample(None, question, context, None, None, None)

    def __call__(self, *texts, **kwargs):
        """
        Args:
            We support multiple use-cases, the following are exclusive:
            X: sequence of SquadExample
            data: sequence of SquadExample
            question: (str, List[str]), batch of question(s) to map along with context
            context: (str, List[str]), batch of context(s) associated with the provided question keyword argument
        Returns:
            dict: {'answer': str, 'score": float, 'start": int, "end": int}
            answer: the textual answer in the intial context
            score: the score the current answer scored for the model
            start: the character index in the original string corresponding to the beginning of the answer' span
            end: the character index in the original string corresponding to the ending of the answer' span
        """
        # Set defaults values
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("doc_stride", 128)
        kwargs.setdefault("max_answer_len", 15)
        kwargs.setdefault("max_seq_len", 384)
        kwargs.setdefault("max_question_len", 64)
        kwargs.setdefault("nthreads", 24)
        kwargs.setdefault("validator", False)
        kwargs.setdefault("batchsize", 48)
        
        if kwargs["topk"] < 1:
            raise ValueError("topk parameter should be >= 1 (got {})".format(kwargs["topk"]))

        if kwargs["max_answer_len"] < 1:
            raise ValueError("max_answer_len parameter should be >= 1 (got {})".format(kwargs["max_answer_len"]))

        # Convert inputs to features
        examples = self._args_parser(*texts, **kwargs)
        
        threads = kwargs['nthreads']
        with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(self.tokenizer,)) as p:
            annotate_ = partial(
                squad_convert_example_to_features,
                max_seq_length=kwargs["max_seq_len"],
                doc_stride=kwargs["doc_stride"],
                max_query_length=kwargs["max_question_len"],
                is_training=False,
            )
            features_list = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                    desc="convert squad examples to features",
                )
            )
            
        all_features = []
        example_index = 0
        for example_features in tqdm(features_list, total=len(features_list), 
                                     desc="add example index and unique id"):
            if not example_features:
                continue
            for example_feature in example_features:
                example_feature.example_index = example_index
                example_feature.unique_id = len(all_features)
                all_features.append(example_feature)
                
            example_index += 1
            
            
        model = nn.DataParallel(self.model).cuda()
        batch_size = kwargs['batchsize']
        
        all_starts, all_ends, all_clss = [], [], []
        for bi in tqdm(range(0, len(all_features), batch_size)):
            batch = all_features[bi: bi + batch_size]
            fw_args = self.inputs_for_model([f.__dict__ for f in batch])
            with torch.no_grad():
                fw_args = {k: torch.tensor(v, device=self.device) for (k, v) in fw_args.items()}
                        
                if kwargs['validator']:
                    fw_args['return_cls'] = True
                    start, end, cls = model(**fw_args)
                    start, end, cls = start.cpu().numpy(), end.cpu().numpy(), cls.cpu().numpy()
                    all_starts.append(start)
                    all_ends.append(end)
                    all_clss.append(cls)
                else:
                    start, end = model(**fw_args)
                    start, end = start.cpu().numpy(), end.cpu().numpy()
                    all_starts.append(start)
                    all_ends.append(end)
                    
        all_starts = np.concatenate(all_starts)
        all_ends = np.concatenate(all_ends)
        
        if len(all_clss) > 0:
            all_clss = np.concatenate(all_clss)
            

        all_answers = []
        for features, example in tqdm(zip(features_list, examples), total=len(examples)):
            fw_args = self.inputs_for_model([f.__dict__ for f in features])

#             # Manage tensor allocation on correct device
#             with self.device_placement():
#                 if self.framework == "tf":
#                     fw_args = {k: tf.constant(v) for (k, v) in fw_args.items()}
#                     start, end = self.model(fw_args)
#                     start, end = start.numpy(), end.numpy()
#                 else:
#                     with torch.no_grad():
#                         # Retrieve the score for the context tokens only (removing question tokens)
#                         fw_args = {k: torch.tensor(v, device=self.device) for (k, v) in fw_args.items()}
                        
#                         if kwargs['validator']:
#                             fw_args['return_cls'] = True
#                             start, end, cls = self.model(**fw_args)
#                             start, end, cls = start.cpu().numpy(), end.cpu().numpy(), cls.cpu().numpy()
#                         else:
#                             start, end = self.model(**fw_args)
#                             start, end = start.cpu().numpy(), end.cpu().numpy()

            sid = features[0].unique_id
        
            start = all_starts[sid: sid + len(features)]
            end = all_ends[sid: sid + len(features)]
                            
            answers = []
            min_null_score = 100000
            min_cls_score = 100000
            for i, (feature, start_, end_) in enumerate(zip(features, start, end)):
                # Normalize logits and spans to retrieve the answer
                start_ = np.exp(start_) / np.sum(np.exp(start_))
                end_ = np.exp(end_) / np.sum(np.exp(end_))

                # Mask padding and question
                start_, end_ = (
                    start_ * np.abs(np.array(feature.p_mask) - 1),
                    end_ * np.abs(np.array(feature.p_mask) - 1),
                )
             
                null_score = (start_[0] * end_[0]).item()
                # Mask CLS
                min_null_score = min(min_null_score, null_score)
                
                start_[0] = end_[0] = 0

                starts, ends, scores = self.decode(start_, end_, kwargs["topk"], kwargs["max_answer_len"])
                char_to_word = np.array(example.char_to_word_offset)
                
                if kwargs['validator']:
                    cls = all_clss[sid: sid + len(features)]
                    cls_ = cls[i]
                    cls_proba = (np.exp(cls_) / np.sum(np.exp(cls_)))[1]
                    
                    min_cls_score = min(min_cls_score, cls_proba)
                    
                    # Convert the answer (tokens) back to the original text
                    answers += [
                        {
                            "score": score.item(),
                            "cls_score": cls_proba,
                            "null_score": null_score,
                            "start": np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
                            "end": np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
                            "answer": remove_space_between_chinese(" ".join(
                                example.doc_tokens[feature.token_to_orig_map[s] : feature.token_to_orig_map[e] + 1]
                            )),
                        }
                        for s, e, score in zip(starts, ends, scores) if score > 0
                    ]
                else:
                    # Convert the answer (tokens) back to the original text
                    answers += [
                        {
                            "score": score.item(),
                            "null_score": null_score,
                            "start": np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
                            "end": np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
                            "answer": remove_space_between_chinese(" ".join(
                                example.doc_tokens[feature.token_to_orig_map[s] : feature.token_to_orig_map[e] + 1]
                            )),
                        }
                        for s, e, score in zip(starts, ends, scores) if score > 0
                    ]
                
            if kwargs['validator']:
                answers.append({
                    "score": min_null_score,
                    "cls_score": min_cls_score,
                    "null_score": min_null_score,
                    "start": 0,
                    "end": 0,
                    "answer": ""
                })
            else:
                answers.append({
                    "score": min_null_score,
                    "null_score": min_null_score,
                    "start": 0,
                    "end": 0,
                    "answer": ""
                })
            
            answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: kwargs["topk"]]
            all_answers.append(answers)

        if len(all_answers) == 1:
            return all_answers[0]
        return all_answers

    def decode(self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int) -> Tuple:
        """
        Take the output of any QuestionAnswering head and will generate probalities for each span to be
        the actual answer.
        In addition, it filters out some unwanted/impossible cases like answer len being greater than
        max_answer_len or answer end position being before the starting position.
        The method supports output the k-best answer through the topk argument.

        Args:
            start: numpy array, holding individual start probabilities for each token
            end: numpy array, holding individual end probabilities for each token
            topk: int, indicates how many possible answer span(s) to extract from the model's output
            max_answer_len: int, maximum size of the answer to extract from the model's output
        """
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
        return start, end, candidates[0, start, end]

    def span_to_answer(self, text: str, start: int, end: int):
        """
        When decoding from token probalities, this method maps token indexes to actual word in
        the initial context.

        Args:
            text: str, the actual context to extract the answer from
            start: int, starting answer token index
            end: int, ending answer token index

        Returns:
            dict: {'answer': str, 'start': int, 'end': int}
        """
        words = []
        token_idx = char_start_idx = char_end_idx = chars_idx = 0

        for i, word in enumerate(text.split(" ")):
            token = self.tokenizer.tokenize(word)

            # Append words if they are in the span
            if start <= token_idx <= end:
                if token_idx == start:
                    char_start_idx = chars_idx

                if token_idx == end:
                    char_end_idx = chars_idx + len(word)

                words += [word]

            # Stop if we went over the end of the answer
            if token_idx > end:
                break

            # Append the subtokenization length to the running index
            token_idx += len(token)
            chars_idx += len(word) + 1

        # Join text with spaces
        return {
            "answer": " ".join(words),
            "start": max(0, char_start_idx),
            "end": min(len(text), char_end_idx),
        }

# +
# from transformers import AutoModelForQuestionAnswering, BertTokenizer, AutoConfig
# from my_model import MyAlbertForQuestionAnswering
# # from question_answering_pipeline import QuestionAnsweringPipeline
# import torch
# import re

# +
# model = MyAlbertForQuestionAnswering.from_pretrained('./models/albert-chinese-large-v2.2/')
# config = AutoConfig.from_pretrained('./models/albert-chinese-large-v2.2/')
# tokenizer = BertTokenizer.from_pretrained('./models/albert-chinese-large-v2.2/')

# +
# question = '上海对什么样的人开展救助管理？'
# context = '123124东方网记者柏可林2月19日报道：记者从上海市城管执法局获悉，疫情期间，城管执法队员细化落实街面防控、社区防控等各项措施，深入开展流浪乞讨人员以及因疫情防控暂无居所来沪人员救助管理工作，加强对摊贩的防控检查、宣传教育和排查登记。'
# example = SquadExample(None, question, context, None, None, None)
# my_squad_convert_example_to_features(example, 512, 128, 60, False)[0].__dict__

# +
# pip = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=2)

# +
# pip(
#     question='上海对什么样的人开展救助管理？',
#     context='东方网记者柏可林2月19日报道：记者从上海市城管执法局获悉，疫情期间，城管执法队员细化落实街面防控、社区防控等各项措施，深入开展流浪乞讨人员以及因疫情防控暂无居所来沪人员救助管理工作，加强对摊贩的防控检查、宣传教育和排查登记。',
#     topk=20, max_answer_len=250
# )
