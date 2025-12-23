
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from generate_prompt import *

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"

class SyntheticGenerator():
    def __init__(self, data: list | None = None, Gen_prompt_question: str = Q_GEN, Gen_prompt_evolving: str = Q_EVOLVE,
                 Gen_prompt_answer: str = A_GEN) -> None:
        self.data = data
        self.q_prompt = Gen_prompt_question
        self.e_prompt = Gen_prompt_evolving
        self.a_prompt = Gen_prompt_answer

    def _generate_question(self, data: list[str]) -> list:
        # Create an OpenAI object

        model = ChatOpenAI(model="glm-4.6",
                         api_key=os.environ.get("ZHIPUAI_API_KEY"),
                         base_url=os.environ.get("ZHIPUAI_BASE_URL"),
                         temperature=0,
                         )
        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant"),
                ("user", self.q_prompt),
            ]
        )

        # Combine the prompt, model, and JsonOutputParser into a chain
        chain = prompt | model

        # Prepare inputs for batch processing
        inputs = [{"context": ctx} for ctx in data]

        results = chain.batch(inputs)
        return [[ctx, result.content] for ctx, result in zip(data, results)]

    def _evolving_question(self, data: list[list[str]]) -> list[list[str]]:
        # Create an OpenAI object
        model = ChatOpenAI(model="glm-4.6",
                           api_key=os.environ.get("ZHIPUAI_API_KEY"),
                           base_url=os.environ.get("ZHIPUAI_BASE_URL"),
                           temperature=0,
                           )

        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant"),
                ("user", self.e_prompt),
            ]
        )

        # Combine the prompt, model, and JsonOutputParser into a chain
        chain = prompt | model

        # Prepare inputs for batch processing
        inputs = [{"context": each[0], "question": each[1]} for each in data]

        results = chain.batch(inputs)
        return [[each[0], result.content] for each, result in zip(data, results)]

    def _generate_answer(self, data: list[list[str]]) -> list[list[str]]:
        # Create an OpenAI object
        model = ChatOpenAI(model="glm-4.6",
                           api_key=os.environ.get("ZHIPUAI_API_KEY"),
                           base_url=os.environ.get("ZHIPUAI_BASE_URL"),
                           temperature=0,
                           )

        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant"),
                ("user", self.a_prompt),
            ]
        )
        # Combine the prompt, model, and JsonOutputParser into a chain
        chain = prompt | model

        # Prepare inputs for batch processing
        inputs = [{"context": ctx, "question": q} for ctx, q in data]

        results = chain.batch(inputs)
        return [[each[0], each[1], result.content] for each, result in zip(data, results)]

    def run(self, data: list | None = None, save_path: str | None = None) -> list:
        raw_data = data if self.data is None else self.data
        print(raw_data)
        if raw_data is None:
            raise ValueError("Empty Data")
        else:
            ctx_data_q = self._generate_question(data=raw_data)
            ctx_data_eq = self._evolving_question(data=ctx_data_q)
            ctx_data_eq_a = self._generate_answer(data=ctx_data_eq)

        if save_path is None:
            return ctx_data_eq_a
        else:
            import pandas as pd
            pd.DataFrame(ctx_data_eq_a, columns=['context', 'question', 'answer']).to_csv(save_path, index=False)
            return ctx_data_eq_a

if __name__ == '__main__':
    sample_context = [
        "（二）投保人在申请投保时，应将被保险人的真实年龄在投保单上填明，如果发生错误应按照下列规定办理：1．投保人申报的被保险人年龄不真实，并且其真实年龄不符合本保险合同约定年龄限制的：保险人可以解除本保险合同，并向投保人退还保险单的现金价值；保险人也可仅对该被保险人（而不是全体被保险人）解除合同，并向投保人退还该被保险人的保险单的现金价值。同时，保险人对于保险合同解除前该被保险人发生的保险事故，不承担赔偿或者给付保险金的责任。 2．投保人申报的被保险人年龄不真实，致使投保人支付的保险费少于应付保险费的，保险人有权更正并要求投保人补交保险费，或者在给付保险金时按照实付保险费与应付保险费的比例支付。 3．投保人申报的被保险人年龄不真实，致使投保人支付的保险费多于应付保险费的，保险人应当将多收的保险费退还投保人。",
        "第二十八条【医院】指保险人与投保人约定的定点医院，未约定定点医院的，则指经中华人民共和国卫生部门评审确定的二级或二级以上的公立医院普通部（不包含公立医院的特需医疗、外宾医疗、干部病房），但不包括主要作为诊所、康复、护理、休养、静养、戒酒、戒毒等或类似的医疗机构。该医院必须具有符合国家有关医院管理规则设置标准的医疗设备，且全天二十四小时有合格医师及护士驻院提供医疗及护理服务。",
        "（一）必选保险责任：单次重度疾病保险金责任在保险期间内，被保险人因意外伤害或在等待期届满日（连续不间断续保为续保生效日）后因意外伤害之外的原因，经符合本保险合同释义约定的医院（以下简称“释义医院”）专科医生确诊初次罹患（指被保险人自出生后首次确诊患有对应疾病，下同）本保险合同约定的任何一组重度疾病中的一种或多种重度疾病，保险人根据本保险合同的约定按照保险单载明的保险金额给付单次重度疾病保险金，给付后本项保险责任终止。本项保险责任下，对于被保险人同时达到本保险合同约定的、两种及以上所属不同组别的重度疾病给付条件，保险人将与被保险人以书面形式协商确定，选择其中一种重度疾病所属组别（确定之后不再接受变更）承担单次重度疾病保险金责任，保险人给付单次重度疾病保险金后本项保险责任终止。",
        "1．跨省异地转诊交通费用保险责任（本保险合同所述“跨省”，指在中华人民共和国境内（不包括港澳台地区）从一个省级行政区（省/自治区/直辖市）到另一个省级行政区，下同）本项责任承保的保险事故包括两类：（1）在保险期间内，被保险人遭受意外伤害事故；（2）自保险期间开始且保险单载明的等待期满之日起（连续不间断续保从续保生效日起），被保险人经释义医院专科医生明确诊断初次发生重大疾病，且该重大疾病不属于责任免除列明的疾病。保险期间内，被保险人因前述两类保险事故在释义医院或保险合同载明的医疗机构接受治疗时，因病情需要跨省转院治疗的，经被保险人申请，并由前述转出释义医院或保险合同载明的医疗机构开具转院证明，对于被保险人发生的合理且必要的因跨省转院治疗产生的公共交通工具费用、救护车使用费，保险人扣除保险单载明的免赔额后按赔付比例给付跨省异地转诊交通费用保险金，最高不得超过跨省异地转诊交通费用保险金额，当累计给付金额达到跨省异地转诊交通费用保险金额时，保险人对被保险人的本项保险责任终止。除另有约定外，被保险人乘坐的飞机舱位级别最高以经济舱（包含超级经济舱）为限，火车（含地铁、轻轨、动车、全列软席列车、其他高速列车）以软卧或一等座为限。如被保险人乘坐的飞机或火车舱位超出上述范围的，保险人核定该项费用损失金额时以上述最高舱位对应金额为准，再根据本保险合同约定扣除保险单载明的免赔额后按赔付比例予以赔偿。"
]

    generator = SyntheticGenerator(data=sample_context)
    synthetic_data = generator.run(save_path='./sample_data.csv')
