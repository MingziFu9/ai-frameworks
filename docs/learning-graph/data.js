// 学习图谱数据：由编者整理，概念与定义取自附录 D 术语表
// 改完后运行 python tools/graph_text.py，检查数据并更新 index.md 里的文字版
var CATS = ["示例类别一", "示例类别二"];
var CHAPTERS = ["", "认识框架", "调用现成模型", "数据与张量", "搭建网络", "训练模型", "训练曲线与调参", "迁移学习", "图像应用", "文本应用", "语音应用", "模型部署", "生态与安全"];
var CHDIRS = ["", "01-intro", "02-pretrained", "03-data", "04-build-network", "05-training", "06-tuning", "07-transfer", "08-image-apps", "09-text-apps", "10-speech-apps", "11-deploy", "12-ecosystem"];
var NODES = [
  {"id": "示例术语", "ch": 1, "cat": 0, "a": "g-示例术语", "d": "这里是术语表中的定义，鼠标悬停时显示"},
  {"id": "另一个术语", "ch": 1, "cat": 1, "a": "g-另一个术语", "d": "以示例术语为基础的概念"}
];
var EDGES = [["示例术语", "另一个术语"]];
