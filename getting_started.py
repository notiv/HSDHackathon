# %%
import numpy as np
import torch
import pandas as pd
from datasets import ClassLabel, Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from pathlib import Path
import pandas as pd


test_size = 0.2
num_classes = 2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
set_class_weights = True

mod_list = Path("huggingface_models.txt").read_text().split("\n")

data_train = pd.read_csv('../data_orig_splits/train_split_0.csv')
data_test = pd.read_csv('../data_orig_splits/test_split_0.csv')

# Split train and test
train = Dataset.from_pandas(pd.DataFrame({'text': data_train["text"], 'label': data_train["label"]}))
test = Dataset.from_pandas(pd.DataFrame({'text': data_test["text"], 'label': data_test["label"]}))

# %% [markdown]
# # Load model and tokenizer
for model_version in mod_list:
    #model_version = 'Hate-speech-CNERG/dehatebert-mono-german'

    # Load tokenizer and model
    tokenizer = 128 # AutoTokenizer.from_pretrained(model_version)
    model = AutoModelForSequenceClassification.from_pretrained(model_version, num_labels=num_classes).to(device)
    max_tokens = model.config.max_position_embeddings

    # %% [markdown]
    # # Tokenize dataset

    # %%
    def tokenize_batch(batch):
        tokens = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=max_tokens)
        return tokens

    train_tokenized = train.map(tokenize_batch, batched=True, batch_size=len(train))
    test_tokenized = test.map(tokenize_batch, batched=True, batch_size=len(test))

    train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # %% [markdown]
    # # Train the model

    # %%
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        return {"f1": f1, "accuracy": acc}



    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=2,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        #warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        evaluation_strategy='steps',
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=1,
        optim='adamw_torch',
        fp16=True,
    #    gradient_accumulation_steps=1,
        learning_rate=5e-6,
        save_strategy='steps',
        save_steps=5000
    )


    # class CustomTrainer(Trainer):
    #     def compute_loss(self, model, inputs, return_outputs=False):
    #         labels = inputs.get("labels")
    #         # forward pass
    #         outputs = model(**inputs)
    #         logits = outputs.get('logits')
    #         # compute custom loss
    #         loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    #         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #         return (loss, outputs) if return_outputs else loss


    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_tokenized,         # training dataset
        eval_dataset=test_tokenized,             # evaluation dataset
        compute_metrics=compute_metrics
    )


    # %%
    trainer.train()


        # %% [markdown]
    t = trainer.evaluate()
    with Path(f"logs/{model_version.replace('/', '_')}").open("a") as f:
        f.write(f"{pd.Timestamp.now()}: {t}")
    torch.save(model.state_dict(), f"final/{model_version.replace('/', '_')}.pth")
