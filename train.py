import torch


images = []
targets = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def move_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return [move_to_device(t, device) for t in x]
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x  # leave other types unchanged

# ===============================  STANDARD TRAINING LOOP  =============================== 
def train_one_epoch(model, dataloader, optimiser, criterion, device):
    model.train()
    running_loss = 0.0   
    n_batches = 0
    
    for images, targets in dataloader:
        images =  move_to_device(images, device)
        targets = move_to_device(targets, device)
        
        optimiser.zero_grad()                # 1) Clear old gradients from previous iteration
        outputs = model(images)              # 2) Forward pass
        loss = criterion(outputs, targets)   # 3) Compute loss
        loss.backward()                      # 4) Backpropagation || Backward pass: Uses the computed loss to calculate the gradients of each parameter (stored in param.grad)
        optimiser.step()                     # 5) Update          || Backward pass: Applies the gradients to update the parameters
    
        running_loss += loss.detach().item()
        n_batches += 1
        # loss =  mean batch loss: loss calculated inside this loop is for the whole batch, not per image 
        #      = Pytorch tensor, part of the computation graph because earlier I called loss.backward()
        #     . detach() -- removes it from the computation graph so that Pytorch stops tracking gradients fo0r it  - This is important — we don’t want to store loss values and keep all the intermediate activations in memory.
        # .item()   -- Converts the 0-dimensional tensor (e.g., tensor(2.345)) into a plain Python float (2.345), which can be summed.

    # running_loss = sum of the mean batch losses over the epoch
    return running_loss / max(n_batches, 1)  # to get the mean batch loss for the epoch 
    
# ======================================================================================================================================================================================== 

 
# ===============================  TORCHVISION TRAINING LOOP  =============================== 
"""
The torchvision models do not use an external criterion. In training they expect:
    * images: List[Tensor[C,H,W]]
    * targets: List[Dict[str, Tensor]] with keys like boxes, labels, etc.
    -- They return a dict of losses you must sum.
"""

def train_one_epoch_torchvision(model, dataloader, optimiser, device):
    model.train()
    running_loss = 0.0
    n_batches = 0
    
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimiser.zero_grad()
        loss_dict = model(images, targets)   # forward returns dict of losses
        # Option B — set a tensor start value for sum()
        loss = sum(loss_dict.values(), torch.tensor(0.0, device=device))
        loss.backward()
        optimiser.step()
        
        running_loss += loss.detach().item()
        n_batches += 1
    
    return running_loss / n_batches, 1
    # return running_loss / max(n_batches, 1)
        
        
"""
Common traps for detection:
    * Don’t stack images; keep them as a list (sizes can differ).
    * Don’t pass a separate criterion; the model already computes it.
    * Ensure every tensor in each target dict is on the same device as the model.
"""


