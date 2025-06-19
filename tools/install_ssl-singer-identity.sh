#/bin/bash

if [ -d "ssl-singer-identity" ]; then
    rm -rf ssl-singer-identity
fi


# # NOTE(jiatong): a versa-specialized implementation for singer identity
git clone https://github.com/ftshijt/ssl-singer-identity.git
cd ssl-singer-identity
pip install -e .
cd ..
