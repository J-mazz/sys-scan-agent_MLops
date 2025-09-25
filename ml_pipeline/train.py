import os
import keras
from keras import layers, ops
from transformers import AutoTokenizer
from datasets import load_from_disk
import numpy as np

# Set Keras backend (can be 'tensorflow', 'jax', or 'torch')
os.environ['KERAS_BACKEND'] = 'tensorflow'  # or 'jax' for better performance on some hardware

# Import backend-specific modules
if keras.backend.backend() == 'tensorflow':
    import tensorflow as tf
    TensorSpec = tf.TensorSpec
    Dataset = tf.data.Dataset
    AUTOTUNE = tf.data.AUTOTUNE
elif keras.backend.backend() == 'jax':
    import jax.numpy as jnp
    # For JAX, we'll need to adapt the dataset creation
else:
    import torch
    # For PyTorch backend

def create_lion_optimizer(learning_rate=2e-4, beta_1=0.9, beta_2=0.99):
    """
    Create Lion optimizer with custom parameters for Keras 3.
    Lion optimizer: https://arxiv.org/abs/2302.06675
    """
    class Lion(keras.optimizers.Optimizer):
        def __init__(self, learning_rate=1e-4, beta_1=0.9, beta_2=0.99, weight_decay=0.0, name="Lion", **kwargs):
            super().__init__(name, learning_rate=learning_rate, **kwargs)
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.weight_decay = weight_decay

        def build(self, var_list):
            super().build(var_list)
            self.momentums = []
            for var in var_list:
                self.momentums.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="momentum"
                    )
                )

        def update_step(self, gradient, variable, momentum):
            lr = ops.cast(self.learning_rate, variable.dtype)

            # Lion update rule
            momentum.assign(self.beta_1 * momentum + (1 - self.beta_1) * gradient)
            update = self.beta_2 * momentum + (1 - self.beta_2) * gradient
            sign_update = ops.sign(update)

            # Apply weight decay
            if self.weight_decay > 0:
                variable.assign_sub(lr * (sign_update + self.weight_decay * variable))
            else:
                variable.assign_sub(lr * sign_update)

        def get_config(self):
            config = super().get_config()
            config.update({
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "weight_decay": self.weight_decay,
            })
            return config

    return Lion(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, weight_decay=0.01)

def preprocess_function(examples, tokenizer, max_length=512):
    """
    Preprocess text data for training.
    """
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf"
    )
    inputs["labels"] = inputs["input_ids"]
    return inputs

def create_tf_dataset(dataset, tokenizer, batch_size=8, max_length=512):
    """
    Create TensorFlow dataset from HuggingFace dataset.
    """
    def generator():
        for example in dataset:
            processed = preprocess_function({"text": example["text"]}, tokenizer, max_length)
            yield {
                "input_ids": processed["input_ids"],
                "attention_mask": processed["attention_mask"],
                "labels": processed["labels"]
            }

    output_signature = {
        "input_ids": TensorSpec(shape=(max_length,), dtype=tf.int32),
        "attention_mask": TensorSpec(shape=(max_length,), dtype=tf.int32),
        "labels": TensorSpec(shape=(max_length,), dtype=tf.int32)
    }

    tf_dataset = Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    tf_dataset = tf_dataset.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)
    return tf_dataset

def train_with_keras():
    """
    Fine-tune LLaMA model using Keras 3 with Lion optimizer on GPU.
    """
    # Set memory growth for GPU (TensorFlow backend)
    if keras.backend.backend() == 'tensorflow':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"❌ GPU memory growth setting failed: {e}")

    dataset_path = "./processed_dataset"
    print(f"Loading pre-processed dataset from {dataset_path}...")
    split_dataset = load_from_disk(dataset_path)
    print("✅ Dataset loaded successfully!")
    print(f"Train samples: {len(split_dataset['train'])}")
    print(f"Validation samples: {len(split_dataset['validation'])}")

    # Model and tokenizer setup
    model_name = "meta-llama/Meta-Llama-3-8B"
    new_model_name = "sys-scan-llama-agent-keras3-lion"

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Create datasets
    batch_size = 4  # Smaller batch size for GPU memory
    max_length = 512

    print("Creating datasets...")
    train_dataset = create_tf_dataset(split_dataset['train'], tokenizer, batch_size, max_length)
    val_dataset = create_tf_dataset(split_dataset['validation'], tokenizer, batch_size, max_length)

    # Load model for Keras 3
    print(f"Loading model {model_name} for Keras 3...")
    try:
        # Try to load as Keras model first
        model = keras.models.load_model(model_name)
    except:
        print("⚠️  Direct Keras model loading failed. Using transformers integration...")
        if keras.backend.backend() == 'tensorflow':
            from transformers import TFAutoModelForCausalLM
            model = TFAutoModelForCausalLM.from_pretrained(
                model_name,
                return_dict=True
            )
        else:
            # For other backends, we'd need different loading logic
            raise NotImplementedError(f"Model loading for {keras.backend.backend()} backend not implemented")

    # Create Lion optimizer
    print("Creating Lion optimizer...")
    optimizer = create_lion_optimizer(learning_rate=2e-4)

    # Compile model with Keras 3
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f"./checkpoints/{new_model_name}_epoch_{{epoch:02d}}",
            save_freq='epoch',
            save_weights_only=True
        ),
        keras.callbacks.TensorBoard(
            log_dir="./logs",
            histogram_freq=1,
            write_graph=True
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    # Create checkpoint directory
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Training configuration
    epochs = 3  # Start with fewer epochs for testing

    print("\n🚀 Starting fine-tuning with Keras and Lion optimizer...")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_length}")
    print(f"Epochs: {epochs}")
    print(f"Optimizer: Lion (lr=2e-4, weight_decay=0.01)")
    print("-" * 50)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Fine-tuning completed!")

    # Save the final model
    print(f"Saving model to {new_model_name}...")
    model.save(new_model_name, save_format='keras')
    tokenizer.save_pretrained(new_model_name)

    print(f"🎉 Model and tokenizer saved to {new_model_name}")

    # Print training summary
    print("\n📊 Training Summary:")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final training accuracy: {history.history['sparse_categorical_accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_sparse_categorical_accuracy'][-1]:.4f}")

    return history

if __name__ == "__main__":
    train_with_keras()