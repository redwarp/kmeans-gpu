# k-means-gpu

Calculate the k average colors in an image using k-means clustering, leveraging your gpu to do the heavy lifting.

Totally 100% inspired by [kmeans-colors](https://github.com/okaneco/kmeans-colors).

## Limitation

Not there yet.

* Only supports RGB color space although Lab looks more promising.
* Currently initialize centroids with random values, should look how complex it is to paralellize kmean++ init.

As this loads an image as a texture to your graphic cards, it also comes with some limitation based on the GPU backends. Like, it probably won't work if the original image is bigger than 4086x4086 pixels.

## Sample

![Tokyo](gfx/tokyo.jpg)

```rust
cargo run --release -- -i .\gfx\tokyo.jpg -k 4
```

![Tokyo with k=4](gfx/tokyo-k4.png)

## Sources

I had to read a bunch of stuff to even start to make sense of it all.
* First of all, the excellent [kmeans-colors](https://github.com/okaneco/kmeans-colors) inspired this project.
* A few articles from [Muthukrishnan](https://muthu.co/):
  + https://muthu.co/reduce-the-number-of-colors-of-an-image-using-uniform-quantization/
  + https://muthu.co/reduce-the-number-of-colors-of-an-image-using-k-means-clustering/
  + https://muthu.co/mathematics-behind-k-mean-clustering-algorithm/
* About prefix sum:
  + https://en.wikipedia.org/wiki/Prefix_sum
  + Prefix sum in wgsl: https://github.com/googlefonts/compute-shader-101/blob/prefix/compute-shader-hello/src/shader.wgsl
  + https://github.com/linebender/piet-gpu/blob/prefix/piet-gpu-hal/examples/shader/prefix.comp
