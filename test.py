from logging import config
from xml.parsers.expat import model
import numpy as np
import time 
import os
from utils.Utils import Utils
from metrics.Metrics import Metrics
from models.spikeresnet import spikeConvNN1, spikeConvNN2, spikeConvNN4, SpikeResNet9Model, SpikeResNet20Model, SpikeResNet, spike_resnet10, spike_resnet18, spike_resnet34, spike_resnet50, spike_resnet101, spike_resnet152
# from models.spikeresnet import SpikeResNet10Model, SpikeResNet18Model  # kept for rollback
from models.plain import spikeLinearNet1
from models.resnet import convNN4, ResNet10, ResNet18, newResNet10Model, newResNet18Model
import torch 
from snntorch import utils
import gc
from utils.experiment_paths import weight_path
from utils.snn_loss import cross_entropy_scores, resolve_ce_options


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def test_accuracy(config, return_configured_accuracy=False):
    '''
    Check accuracy of a trained model on ID test data.

    Spiking models always report spike-count and temporal-mean membrane
    accuracy. When ``return_configured_accuracy`` is true, they additionally
    report accuracy using the same source, temporal mode, and population
    reduction as a configured cross-entropy objective.
    '''

    # Load database
    # print(f"dataSet: {dataSet}")
    dataset_train, dataset_test = Utils.load_data(config.dataset_ID, config)
    testSize = len(dataset_test)
    # print(f"Number of test set images is: {testSize}")

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, config.dataset_ID)

    # Load data
    train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, config.batch_size, config.dataset_ID, True)

    # Get the device
    device = Utils.get_device()
    # device = torch.device("cpu")

    # Get image size based on dataset
    if config.dataset_ID in ['CIFAR10', 'CIFAR100', 'tImage200']:
        feature_size = 32
    elif config.dataset_ID in ['MNIST', 'FMNIST', 'KMNIST']:
        feature_size = 28
    else:
        feature_size = 28  # default

    if config.model_type == 'conv':
        # Define model
        if config.resnet_model == 4:
            model = convNN4(numberOfChannels=channels, 
                            numberOfClasses=config.num_classes, 
                            feature_size=32, 
                            expansion=config.expansion)
        elif config.resnet_model == 10:
            model = ResNet10(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        expansion=config.expansion)
        elif config.resnet_model == 18:
            model = ResNet18(numberOfChannels=channels, 
                                        numberOfClasses=config.num_classes, 
                                        expansion=config.expansion)
        else:
            print("Model not defined for the given ResNet configuration.")
            return -1

        # Load weights
        weightsName = str(weight_path(config))
        if not os.path.exists(weightsName):
            print(f"Weights file not found: {weightsName}")
            return -1

        model.load_state_dict(torch.load(weightsName, weights_only=False))
        model = model.to(device)

        # Evaluate model
        model.eval()
        total_correct_membrane = 0
        total_samples = 0
        with torch.no_grad():
            for batch, labels in test_loader:
                batch = batch.to(device)
                labels = labels.to(device)

                # Forward pass
                mem, _ = model(batch)
                if model.expansion > 1:
                    mem = mem.reshape(mem.shape[0], model.numberOfClasses, model.expansion).sum(dim=2)
                predicted = torch.argmax(mem, dim=1)
                correct = (predicted == labels).float()
                total_correct_membrane += correct.sum()
                total_samples += batch.size(0)

        acc_membrane = total_correct_membrane / total_samples * 100
        return acc_membrane

    elif config.model_type == 'spike':

        # For spiking neural network we need number of steps
        beta = 0.95
        threshold = 0.25

        # Define model
        if config.resnet_model == 1:
            model = spikeConvNN1(numberOfChannels=channels,
                                 numberOfClasses=config.num_classes, 
                                 beta=beta, 
                                 threshold=threshold)
        elif config.resnet_model == 2:
            model = spikeConvNN2(numberOfChannels=channels, 
                                 numberOfClasses=config.num_classes, 
                                 beta=beta, 
                                 threshold=threshold,
                                 feature_size=feature_size)
        elif config.resnet_model == 4:
            model = spikeConvNN4(numberOfChannels=channels, 
                                 numberOfClasses=config.num_classes, 
                                 beta=beta, 
                                 threshold=threshold, 
                                 feature_size=feature_size, 
                                 numberOfSteps=config.num_time_steps_train, 
                                 expansion=config.expansion)
        elif config.resnet_model == 9:
            model = SpikeResNet9Model(numberOfChannels=channels, 
                                      numberOfClasses=config.num_classes, 
                                      beta=beta, 
                                      threshold=threshold)
        # elif config.resnet_model == 10:
        #     model = SpikeResNet10Model(numberOfChannels=channels,
        #                                numberOfClasses=config.num_classes,
        #                                beta=beta, threshold=threshold,
        #                                numberOfSteps=config.num_time_steps_train,
        #                                expansion=config.expansion)
        # elif config.resnet_model == 18:
        #     model = SpikeResNet18Model(numberOfChannels=channels,
        #                                numberOfClasses=config.num_classes,
        #                                beta=beta, threshold=threshold,
        #                                numberOfSteps=config.num_time_steps_train,
        #                                expansion=config.expansion)
        elif config.resnet_model in (10, 18, 34, 50, 101, 152):
            _factory = {
                10:  spike_resnet10,
                18:  spike_resnet18,
                34:  spike_resnet34,
                50:  spike_resnet50,
                101: spike_resnet101,
                152: spike_resnet152,
            }[config.resnet_model]
            model = _factory(numberOfChannels=channels,
                             numberOfClasses=config.num_classes,
                             beta=beta,
                             threshold=threshold,
                             numberOfSteps=config.num_time_steps_train,
                             expansion=config.expansion)
        elif config.resnet_model == 20:
            model = SpikeResNet20Model(numberOfChannels=channels,
                                       numberOfClasses=config.num_classes,
                                       beta=beta,
                                       threshold=threshold,
                                       numberOfSteps=config.num_time_steps_train,
                                       expansion=config.expansion)
        elif config.resnet_model == 21:
            model = spikeLinearNet1(numberOfChannels=channels, numberOfClasses=config.num_classes, beta=beta, threshold=threshold)
        else:
            print("Model Not defined")
            return -1
        
        # Load weights
        model = model.to(device)
        torch.cuda.empty_cache()

        weightsName = str(weight_path(config))
        if not os.path.exists(weightsName):
            # try alternate filename without the loss token (legacy name)
            wN = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed)+'.pth'
            if os.path.exists(wN):
                print(f"Found legacy weights file: {wN}, renaming to new format.")
                try:
                    os.rename(wN, weightsName)
                    print(f"Renamed legacy weights file {wN} -> {weightsName}")
                except Exception as e:
                    print(f"Warning: failed to rename {wN} -> {weightsName}: {e}")
            else:
                print(f"Weights file not found: {weightsName}")
                return -1, -1
        # print(f"weightsName: {weightsName}")
        raw_sd = torch.load(weightsName, weights_only=False)
        model.load_state_dict(SpikeResNet.remap_legacy_state_dict(raw_sd))
        model = model.to(device)

        # print("Check test images.")
        testLen = 0
        i = 0
        # print(f"Number of batches: {len(test_loader)}")
        model.eval()
        
        total_correct_spikes = 0
        total_correct_membrane = 0
        total_correct_configured = 0
        total_samples = 0
        uses_cross_entropy = config.loss == "cross_entropy"
        if uses_cross_entropy:
            ce_source, ce_mode, population_reduction = resolve_ce_options(
                getattr(config, "ce_source", None),
                getattr(config, "ce_mode", None),
                getattr(config, "population_reduction", None),
                config.fit,
            )
        else:
            ce_source = "spikes"
            ce_mode = "spike_count"
            population_reduction = "sum"
        with torch.no_grad():
            for batch, labels in test_loader:
                testLen += len(batch)
                batch = batch.to(device)
                la = labels[0].item()
                labels = labels.to(device)
                # model.mem1 = final_membranes['mem1'].to(device)
                # model.mem2 = final_membranes['mem2'].to(device)
                # model.mem3 = final_membranes['mem3'].to(device)
                # model.mem4 = final_membranes['mem4'].to(device)
                # model.mem5 = final_membranes['mem5'].to(device)
                # Generate predictions/ forward pass
                outputs = model(
                    batch,
                    config.num_time_steps_extract,
                    return_logits_trace=uses_cross_entropy and ce_source == "logits",
                )
                spikes, _, membrane, _ = outputs[:4]
                # print(f"spikes.max: {spikes.max()}")
                # l_spikes = loss_fn(spikes, labels)
                # testLossSpikes.append(l_spikes.item())
                # a1, a2 = model.accuracy_spike(model, numberOfSteps, batch, labels, device)
                # print(f"testLen: {testLen}, a1: {a1}, {a1.item()}, a2: {a2}, {a2.item()}")
                # testAcc[0, i] = a2.item()
                # testAccList.append(a1.item())
                # On spikes
                # print(f"spikes.shape: {spikes.shape}, labels.shape: {labels.shape}")
                batch_size = batch.size(0)
                spike_scores = cross_entropy_scores(
                    spikes,
                    mode="spike_count",
                    num_classes=model.numberOfClasses,
                    population_reduction="sum",
                )
                total_correct_spikes += (
                    spike_scores.argmax(dim=1) == labels
                ).sum().item()

                # On membrane
                membrane_scores = cross_entropy_scores(
                    membrane,
                    mode="temporal_mean",
                    num_classes=model.numberOfClasses,
                    population_reduction="sum",
                )
                total_correct_membrane += (
                    membrane_scores.argmax(dim=1) == labels
                ).sum().item()

                traces = {"spikes": spikes, "membranes": membrane}
                if uses_cross_entropy and ce_source == "logits":
                    if len(outputs) != 5:
                        raise ValueError(
                            f"{type(model).__name__} did not return requested logits."
                        )
                    traces["logits"] = outputs[4]
                configured_scores = cross_entropy_scores(
                    traces[ce_source],
                    mode=ce_mode,
                    num_classes=model.numberOfClasses,
                    population_reduction=population_reduction,
                )
                total_correct_configured += (
                    configured_scores.argmax(dim=1) == labels
                ).sum().item()

                total_samples += batch_size
                
                print(f"\rCurrent: {total_correct_spikes}, Test accuracy on spikes: {total_correct_spikes/total_samples*100:05.2f}, Test accuracy on membrane: {total_correct_membrane/total_samples*100:05.2f}, progress: {i+1}/{len(test_loader)}", end='', flush=True)
                i += 1

        print("\nDone.")
        acc_membrane = total_correct_membrane / total_samples * 100
        acc_spikes = total_correct_spikes / total_samples * 100
        acc_configured = total_correct_configured / total_samples * 100

        del model 
        del train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
        if return_configured_accuracy:
            return acc_spikes, acc_membrane, acc_configured
        return acc_spikes, acc_membrane

