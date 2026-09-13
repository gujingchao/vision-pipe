# Docker

```bash
docker build -f docker/Dockerfile -t vision-pipe .
docker run --rm -v $PWD/examples/images:/data -v $PWD/out:/out vision-pipe \
  infer --input /data --output /out --save-json
```
