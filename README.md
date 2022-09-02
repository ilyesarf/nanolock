Smart screen locking with Facial Verification.


## Requirements

- Python3
- <a href="https://www.tensorflow.org/install/pip"> Tensorflow </a> (Python3)
- Gmail account
- BASH (as a default shell)

## Installation

```
$ git clone https://github.com/Quimzy/nanolock.git

$ pip3 install requirements.txt

$ chmod +x setup.sh

$ sudo ./setup.sh
```

## Setup Gmail for Alerts

1. <a href="https://support.google.com/accounts/answer/185833?hl=en"> Generate & Copy App Password </a>
2. Run ./setup.sh
3. Set credentials (put App Password as password)

## TODO

- [X] Facial Verification
- [X] Lock screen if user is not detected
- [X] Alert user (probably through email)