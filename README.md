# tf serving + object_detection 测试

# 1. 安装环境

```
pip install -r requirements.txt
```

# 2. 使用 tf_serving 部署 object_detection 服务

``` 
chmod +x run_server.sh && ./run_server.sh
```

# 3. grpc 调用 tf_serving 服务

``` 
PYTHONPATH=$(pwd) python tf_serving_api.py
```
