import torch

__all__ = ['train']


def train_one_epoch(epoch_index, model, loss_fn, opt, train_loader, device='cpu'):
    running_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.float().to(device)
        y = y.to(device)

        opt.zero_grad()
        outputs = model(x)

        l = loss_fn(outputs, y)
        l.backward()

        opt.step()

        running_loss += l.item()

        if i % 10 == 0:
            print(f"Epoch: {epoch_index} batch: {i} Loss: {running_loss}")
    return running_loss / (i + 1)


def train(model, loss_fn, train_loader, test_loader, device='cpu', epochs=500):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_loss = 1
    for epoch in range(epochs):
        model.train(True)

        loss = train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device)

        model.eval()
        running_loss = 0
        for i, vdata in enumerate(test_loader):
            x, y = vdata
            x = x.float().to(device)
            y = y.float().to(device)
            output = model(x.to(device))
            guess = torch.argmax(output, axis=1)
            running_loss += torch.sum(guess == y).item() / 8



        avg_vloss = running_loss / (i + 1)
        print(f"LOSS train {loss} test {avg_vloss}")

        if 1 - avg_vloss < best_loss:

            print("saving model")
            best_loss = 1 - avg_vloss
            torch.save(model.state_dict(), "checkpoint.pt")
