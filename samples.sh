#!/bin/sh

cargo run --release -- reduce -i ./gfx/tokyo.png -c 8
cargo run --release -- reduce -i ./gfx/tokyo.png -c 8 -m dither
cargo run --release -- palette -i ./gfx/tokyo.png -c 8 -s 40
cargo run --release -- find -i ./gfx/tokyo.png -p "#050505,#ffffff,#ff0000" -o ./gfx/tokyo-find-replace-dark-white-red.png
cargo run --release -- find -i ./gfx/tokyo.png -p "#050505,#ffffff,#ff0000" -m dither -o ./gfx/tokyo-find-dither-dark-white-red.png
cargo run --release -- find -i ./gfx/tokyo.png -p ./gfx/apollo-1x.png -m dither -o ./gfx/tokyo-find-dither-apollo.png

oxipng -r ./gfx
