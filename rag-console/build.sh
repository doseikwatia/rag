export CC=gcc-12
export CXX=g++-12
export CUDAHOSTCXX=g++-12
cargo build --release
cp target/release/rag-console ../bin