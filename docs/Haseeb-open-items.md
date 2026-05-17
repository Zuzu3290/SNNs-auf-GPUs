# To DO

- different prams for different frameworks like Threshold vs v_th  in snn_torch vs norse. Implement them either in config as frameworks blocks or somre mapping , example beta has range 0-1, and tau: 10-1000 Hz even they have same meaning, found beta nad tau as clashing so far.

---

- No loss function in Norse:
Norse has no built-in loss functions — it only provides neuron dynamics (LIF, etc.) and leaves loss entirely to you. So the options are:

Option 1 — Keep SF.mse_count_loss (what we did)
It's purely tensor math operating on [T, B, num_classes] — it doesn't use any SNNTorch neuron internals. Norse produces the same shaped spike tensor, so it works fine.

Option 2 — Standard PyTorch CrossEntropy on summed spikes


loss_fn = torch.nn.CrossEntropyLoss()
# usage: loss_fn(spk_rec.sum(0), targets)  # sum over T → [B, num_classes]
For this project specifically, Option 1 is actually the right choice. You're comparing frameworks — if both models use the exact same loss function, you isolate the variable to the neuron dynamics only. Switching loss functions between SNNTorch and Norse would make the comparison unfair.

---





# Improevements in Future




# Found Issue --- Status