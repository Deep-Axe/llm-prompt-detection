"""Train the planned distilbert model with the shared runner."""
import sys
from train_neural import main

if __name__ == '__main__':
    sys.argv.insert(1,'distilbert')
    main()
