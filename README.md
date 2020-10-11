# Fast DeHz

Fast DeHz for real-time low-light enhancement

![Results of different enhancing methods](/misc/demo.png)

## How to use

```
python flow.py [video] [--fast]
```

If you want to evaluate on a single frame

```
python dehz.py [image] [--fast]
```

Approximated pointwise operation

```
python pointwise.py [image]
```

## Performance

Settings

* Frame size: 640x360
* CPU: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz

| Method    | Avg. Runtime per Frame | FPS   |
| --------- | ---------------------- | ----- |
| DeHz      | 0.15923                | 6.28  |
| Fast DeHz | 0.02292                | 43.63 |
