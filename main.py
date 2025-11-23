import time
import os
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig
from codecarbon import EmissionsTracker
from hydra.utils import get_original_cwd

from src import utils
import wandb


def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)
    best_val_acc = 0.0
    
    # Store per-epoch metrics
    epoch_metrics = []

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels) # push to GPU

            # print("input shape:",inputs['sample'].shape)
            # print("label shape:",labels['class_labels'].shape)
            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        epoch_time = time.time() - start_time
        start_time = time.time()
        
        # Store epoch metrics
        epoch_data = {
            "epoch": epoch,
            "train_metrics": dict(train_results),
            "train_time_seconds": epoch_time,
            "val_metrics": None
        }

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            val_results = validate_or_test(opt, model, "val", epoch=epoch, best_val_acc=best_val_acc)
            best_val_acc = val_results["best_val_acc"]
            epoch_data["val_metrics"] = val_results["metrics"]
        
        epoch_metrics.append(epoch_data)

    return model, epoch_metrics


def validate_or_test(opt, model, partition, epoch=None, best_val_acc=1.0):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            scalar_outputs = model.forward_downstream_multi_pass(
                inputs, labels, scalar_outputs=scalar_outputs
            )
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    # save model if classification accuracy is better than previous best
    if test_results["classification_accuracy"] > best_val_acc:
        print("saving model")
        best_val_acc = test_results["classification_accuracy"]
        utils.save_model(model)

    model.train()
    return {"best_val_acc": best_val_acc, "metrics": dict(test_results)}


@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    
    # Setup emissions tracking directory
    emissions_dir = os.path.join(get_original_cwd(), "emissions")
    os.makedirs(emissions_dir, exist_ok=True)
    
    # Get goodness function name from config
    goodness_function = getattr(opt.model, 'goodness_function', 'sum_of_squares')
    
    # Initialize codecarbon tracker
    tracker = EmissionsTracker(
        project_name=f"FF_{opt.input.dataset}_{goodness_function}",
        output_dir=emissions_dir,
        output_file=f"{opt.input.dataset}_{goodness_function}_emissions.csv",
        log_level="warning"
    )
    
    # Initialize WandB in offline mode to avoid connection issues
    run = wandb.init(
        project="FF",
        entity=None,
        name=None,
        mode="offline",  # Run in offline mode - logs saved locally
        reinit=False,
        config=dict(opt)
    )
    
    # Start carbon tracking
    tracker.start()
    experiment_start_time = time.time()
    
    try:
        # Train the model
        model, optimizer = utils.get_model_and_optimizer(opt)
        model, epoch_metrics = train(opt, model, optimizer)
        
        # Get final validation results
        print("\n" + "="*60)
        print("FINAL VALIDATION")
        print("="*60)
        final_val_results = defaultdict(float)
        val_loader = utils.get_data(opt, "val")
        num_val_steps = len(val_loader)
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = utils.preprocess_inputs(opt, inputs, labels)
                scalar_outputs = model.forward_downstream_classification_model(inputs, labels)
                scalar_outputs = model.forward_downstream_multi_pass(inputs, labels, scalar_outputs=scalar_outputs)
                final_val_results = utils.log_results(final_val_results, scalar_outputs, num_val_steps)
        
        # Get test results if enabled
        final_test_results = None
        if opt.training.final_test:
            print("\n" + "="*60)
            print("FINAL TEST")
            print("="*60)
            final_test_results = defaultdict(float)
            test_loader = utils.get_data(opt, "test")
            num_test_steps = len(test_loader)
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = utils.preprocess_inputs(opt, inputs, labels)
                    scalar_outputs = model.forward_downstream_classification_model(inputs, labels)
                    scalar_outputs = model.forward_downstream_multi_pass(inputs, labels, scalar_outputs=scalar_outputs)
                    final_test_results = utils.log_results(final_test_results, scalar_outputs, num_test_steps)
        
        # Calculate total training time
        total_training_time = time.time() - experiment_start_time
        
    finally:
        # Stop carbon tracking
        emissions = tracker.stop()
        
        # Get carbon data
        emissions_file = os.path.join(emissions_dir, f"{opt.input.dataset}_{goodness_function}_emissions.csv")
        carbon_data = utils.get_carbon_summary(emissions_file)
        
        # Save structured results
        train_results = {"note": "Training metrics logged during training"}
        results_filename = utils.save_experiment_results(
            opt=opt,
            train_results=train_results,
            val_results=dict(final_val_results) if 'final_val_results' in locals() else {},
            test_results=dict(final_test_results) if final_test_results else {},
            training_time=total_training_time if 'total_training_time' in locals() else 0,
            carbon_data=carbon_data
        )
        
        # Save per-epoch metrics
        if 'epoch_metrics' in locals():
            utils.save_epoch_metrics(opt, epoch_metrics, results_filename)
        
        # Print carbon footprint summary
        if carbon_data:
            print("\n" + "="*60)
            print("CARBON FOOTPRINT SUMMARY")
            print("="*60)
            print(f"Total CO2 Emissions: {carbon_data['emissions_g_co2']:.4f} g CO2")
            print(f"Energy Consumed: {carbon_data['energy_consumed_kwh']:.6f} kWh")
            print(f"Duration: {carbon_data['duration_seconds']:.2f} seconds")
            print(f"Country: {carbon_data['country']}")
            print("="*60 + "\n")
        
        run.finish()


if __name__ == "__main__":
    my_main()
