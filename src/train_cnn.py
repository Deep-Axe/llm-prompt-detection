"""Train the planned CNN with the shared PyTorch runner."""
import sys
from train_neural import main

if __name__ == '__main__':
    sys.argv.insert(1,'cnn')
    main()
