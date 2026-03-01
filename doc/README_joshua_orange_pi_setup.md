Installed [Ubuntu 24.04 from Joshua](https://github.com/Joshua-Riek/ubuntu-rockchip/)

Followed [Axeleras Device-Tree Fix](https://support.axelera.ai/hc/en-us/articles/27059519168146-Bring-up-Voyager-SDK-in-Orange-Pi-5-Plus)

Followed this [post for gstreamer fix](https://community.axelera.ai/project-challenge-27/missing-gstreamer-plugins-on-arm64-rk3588-root-cause-solution-1104?postid=2373#post2373)


```
sudo apt-get update
sudo apt install -y librga-dev
```

Installed packages mentioned in [Axeleras Bring-Up Guide for NanoPC-T6](https://support.axelera.ai/hc/en-us/articles/31859388491794-Bring-up-Voyager-SDK-in-NanoPC-T6)

```
sudo apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools
```

Then build:
```
source venv/bin/activate
make clobber-libs && make operators
```