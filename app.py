import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ctransformers import CTransformers
from langchain_ollama import OllamaLLM

def getLLMResponse(form_input, email_sender, email_recipient, language):
    '''
    ● name   : getLLMResponse
    ● description : getLLMResponse 함수는 주어진 입력을 사용해서, LLM(대형 언어 모델)로부터 이메일 응답을 생성합니다.
    ● parameters :
        - form_input: 사용자가 입력한 이메일 주제.
        - email_sender: 이메일을 보낸 사람의 이름.
        - email_recipient: 이메일을 받는 사람의 이름.
        - language: 이메일이 생성될 언어 (한국어 또는 영어).
    ● 반환값 : LLM이 생성한 이메일 응답 텍스트.
    '''
    
    # 압축된 AI 모델 불러오기 (2가지 방법 중 하나 선택)
    # 1순위. Ollama 라이브러리로 "llama3.1:8b" 모델 불러오기
    llm = OllamaLLM(model="llama3.1:8b", temperature=0.7)
    
    # 사용 환경에 따라 아래 코드로 변경 가능
    # 2순위. CTransformers 라이브러리로 "llama-2-7b-chat" 모델 불러오기
    # llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",
    #                     model_type="llama",
    #                     config={
    #                         'temperature': 0.01,
    #                         'max_new_tokens': 512
    #                     }
    #                     )
    
    # 이메일 템플릿 생성
    if language == "한국어":
        template = '''
        당신은 전문적인 이메일 작성자입니다. 
        주제 "{email_topic}"를 포함한 이메일을 작성해주세요.
        \n\n보낸 사람 : {sender_name}
        \n받는 사람 : {recipient_name}
        전부 {language}로 번역해서 작성해 주세요. 한문은 내용에서 제외해주세요.
        \n\n이메일 내용:
        '''
    else:
        template = '''
        Write an email including the topic {email_topic}.
        \n\nSender: {sender_name}
        \nRecipient: {recipient_name} 
        Please write the entire email in {language}.
        \n\nEmail content:
        '''
    
    # PromptTemplate 객체 생성
    prompt = PromptTemplate(
        input_variables=["email_topic", "sender_name", "recipient_name", "language"],
        template=template
    )
    
    # 프롬프트 포맷팅 및 LLM 호출
    formatted_prompt = prompt.format(
        email_topic=form_input,
        sender_name=email_sender,
        recipient_name=email_recipient,
        language=language
    )
    
    # LLM으로부터 응답 받기
    response = llm.invoke(formatted_prompt)
    
    return response


# Streamlit UI 구성
st.title("📧 이메일 자동 생성기")
st.markdown("---")

# 이메일 작성 언어 선택
language_choice = st.selectbox('이메일을 작성할 언어를 선택하세요:', ['한국어', 'English'])

# 이메일 주제 입력란
form_input = st.text_area('이메일 주제를 입력하세요', height=100)

# 발신자와 수신자 입력란
col1, col2 = st.columns([10, 10])
with col1:
    email_sender = st.text_input('보낸 사람 이름')
with col2:
    email_recipient = st.text_input('받는 사람 이름')

submit = st.button("생성하기")

# '생성하기' 버튼이 클릭되면, 아래 코드를 실행합니다.
if submit:
    if not form_input or not email_sender or not email_recipient:
        st.warning("모든 필드를 입력해주세요!")
    else:
        with st.spinner('생성 중입니다...'):
            response = getLLMResponse(form_input, email_sender, email_recipient, language_choice)
            st.success("이메일이 생성되었습니다!")
            st.markdown("---")
            st.write(response)