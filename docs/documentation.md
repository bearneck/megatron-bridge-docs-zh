# 文档开发

## 构建文档

以下部分描述了如何设置和构建 Megatron Bridge 文档。

切换到文档源文件夹并生成 HTML 输出。

```sh
cd docs/
uv run --only-group docs sphinx-build . _build/html
```

* 生成的 HTML 文件位于项目 `docs/` 文件夹下创建的 `_build/html` 文件夹中。
* 生成的 Python API 文档放置在 `docs/` 文件夹下的 `apidocs` 目录中。

```{注意}
如果遇到错误 "Failed to generate package metadata for megatron-core @ directory+3rdparty/Megatron-LM,"，请运行以下命令来安装必要的子模块：

`git submodule update --init --recursive`
```

## 实时构建

在编写文档时，提供一个能实时更新的文档服务会很有帮助。

为此，请运行：

```sh
cd docs/
uv run --only-group docs sphinx-autobuild . _build/html --port 12345 --host 0.0.0.0
```

打开网页浏览器并访问 `http://${运行SPHINX命令的主机}:12345` 来查看输出。

## 在 Python 文档字符串中编写测试

任何带有 `{doctest}` 指令的三重反引号代码块中的代码都将被测试。格式遵循 Python 的 doctest 模块语法，其中 `>>>` 表示 Python 输入，下一行显示预期输出。以下是一个示例：

```python
def add(x: int, y: int) -> int:
    """
    将两个整数相加。

    参数：
        x (int): 要相加的第一个整数。
        y (int): 要相加的第二个整数。

    返回：
        int: x 和 y 的和。

    示例：
    ```{doctest}
    >>> from megatron.bridge.made_up_package import add
    >>> add(1, 2)
    3
    ```

    """
    return x + y
```

## 运行 Python 文档字符串中的测试

您可以使用以下命令运行 Python 文档字符串中的测试：

```sh
cd docs/
uv run --only-group docs sphinx-build -b doctest . _build/doctest
```

## 文档版本

以下三个文件控制版本切换器。在尝试发布新版本的文档之前，请更新 `docs/` 文件夹中的这些文件以匹配最新的版本号。

发布新版本的文档时，请将 ``version`` 和 ``release`` 变量与您的 GitHub 发布版本保持一致。

```{危险}
`latest` 应仅为主分支中的 ``version`` 和 ``release`` 变量。
```

### versions1.json

此 JSON 文件定义了切换器下拉菜单中显示的版本。向 JSON 添加新版本时，请确保 ``version`` 和 ``url`` 包含与您的发布版本相同的版本号。

示例：

```{code-block} json
:caption: 示例：versions1.json 
:emphasize-lines: 9,10

[
    {
        "preferred": true,
        "version": "latest",
        "url": "https://docs.nvidia.com/nemo/megatron-bridge/latest/"
    },
    {

        "version": "#.#.#",
        "url": "https://docs.nvidia.com/nemo/megatron-bridge/#.#.#/"
    },
    {

        "version": "0.1.0",
        "url": "https://docs.nvidia.com/nemo/megatron-bridge/0.1.0/"
    }
]
```

```{提示}
为 ``url`` 变量使用绝对 URL。
```

## project.json

此 JSON 文件告诉版本切换器，文档与切换器中选择的版本匹配。``version`` 应包含与您的发布版本相同的版本号。

```{code-block} json
:caption: 示例：project.json 
:emphasize-lines: 3

{
    "name": "megatron-bridge",
    "version": "#.#.#"
}
```

## conf.py

conf.py 中的 ``release`` 应包含与您的发布版本相同的版本号。

```{code-block} python
:caption: 示例：conf.py 
:emphasize-lines: 7

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Megatron Bridge"
copyright = "2025, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "#.#.#"
```