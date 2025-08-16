from unstructured.partition.pdf import partition_pdf

pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructuredde pdf模块加载并解析PDF文档
#strategy 参数控制将用于处理 PDF 的方法。可用的 PDF 策略有 "auto" 、 "hi_res" 、 "ocr_only" 和 "fast" 。
#"auto" 策略将根据文档特征和函数关键字参数选择分区策略。如果 skip_infer_table_types 设置为空列表，则策略将为 "hi_res"，因为这是目前唯一能够提取 PDF 表格的策略。
#"hi_res" 策略将使用 detectron2_onnx 识别文档布局。 "hi_res" 的优势在于它利用文档布局来获取关于文档元素的额外信息。
elements = partition_pdf(filename=pdf_path,skip_infer_table_types=False,strastrategy='hi_res')

#统计解析结果并打印
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

#统计elements中的元素类型
element_types = [type(e).__name__ for e in elements]
print(f"元素类型: {set(element_types)}")

#显示所有元素
for i, element in enumerate(elements):
    print(f"元素 {i+1}: {element}")
    print("-" * 60)

tables = [el for el in elements if el.category == "Table"]
if tables:
    print(f"pdf中的表格文字有：\n{tables[0].text}")
    print(tables[0].metadata.text_as_html)
else:
    print("未检测到表格元素。")
print(tables[0].metadata.text_as_html)