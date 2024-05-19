import torch
import os
from Levenshtein import distance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.open_source_functionality import load_os_dataset, plot_history, try_model

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

BATCH_SIZE = 16
EPOCHS = 1

model_list = ["microsoft/trocr-small-stage1", "microsoft/trocr-base-stage1"]
base_path = "models/trocr/"
dataset_path = 'dataset/transfer_dataset/'
train_dataset_path = os.path.join(dataset_path, 'train')
val_dataset_path = os.path.join(dataset_path, 'val')

for model_name in model_list:
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = TrOCRProcessor.from_pretrained(model_name)
    save_model_name = os.path.join(base_path, model_name.split('/')[1])
    
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_gram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    train_dataset, eval_dataset = load_os_dataset(train_dataset_path, val_dataset_path, processor)

    training_args = Seq2SeqTrainingArguments(  
        predict_with_generate=True,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=False, 
        output_dir=save_model_name,
        logging_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="levenshtein",  
        greater_is_better=False  
    )

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        sum_leven = 0
        for label, pred in zip(label_str, pred_str):
            sum_leven += distance(label, pred)
        levenshtein = sum_leven / len(label_str)
    
        return {"levenshtein": levenshtein}
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.image_processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    plot_history(trainer, save_model_name, show_plot=False)
    trainer.save_model(save_model_name)
    #try_model(save_model_name, processor, model_name, eval_dataset)
    
print("Training complete!")
