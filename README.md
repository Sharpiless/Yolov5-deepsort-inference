本文地址：[https://blog.csdn.net/weixin_44936889/article/details/112002152](https://blog.csdn.net/weixin_44936889/article/details/112002152)

# 注意：

本项目使用Yolov5 3.0版本，4.0版本需要替换掉models和utils文件夹

# 项目简介：
使用YOLOv5+Deepsort实现车辆行人追踪和计数，代码封装成一个Detector类，更容易嵌入到自己的项目中。

代码地址（欢迎star）：

[https://github.com/Sharpiless/Yolov5-deepsort-inference](https://github.com/Sharpiless/Yolov5-deepsort-inference)

最终效果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201231090541223.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
# YOLOv5检测器：

```python
class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):

        self.weights = 'weights/yolov5m.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        # torch.save(model, 'test.pt')
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)

        pred_boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in ['person', 'car', 'truck']:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes

```

调用 self.detect 方法返回图像和预测结果

# DeepSort追踪器：

```python
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
```

调用 self.update 方法更新追踪结果

# 运行demo：

```bash
python demo.py
```

# 训练自己的模型：
参考我的另一篇博客：

[【小白CV】手把手教你用YOLOv5训练自己的数据集（从Windows环境配置到模型部署）](https://blog.csdn.net/weixin_44936889/article/details/110661862)

训练好后放到 weights 文件夹下

# 调用接口：

## 创建检测器：

```python
from AIDetector_pytorch import Detector

det = Detector()
```

## 调用检测接口：

```python
func_status = {}
func_status['headpose'] = None

result = det.feedCap(im, func_status)
```

其中 im 为 BGR 图像

返回的 result 是字典，result['frame'] 返回可视化后的图像

# 关注我的公众号：

感兴趣的同学关注我的公众号——可达鸭的深度学习教程：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210127153004430.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)


# 联系作者：

> B站：[https://space.bilibili.com/470550823](https://space.bilibili.com/470550823)

> CSDN：[https://blog.csdn.net/weixin_44936889](https://blog.csdn.net/weixin_44936889)

> AI Studio：[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156)

> Github：[https://github.com/Sharpiless](https://github.com/Sharpiless)

遵循 GNU General Public License v3.0 协议，标明目标检测部分来源：https://github.com/ultralytics/yolov5/
