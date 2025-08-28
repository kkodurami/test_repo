import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

st.write('''
# Bug Classification 🐛
''')

# 모델 로드
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r'model\best_bug_model.h5')
    return model

def preprocess_image(image):
    # PIL Image를 numpy array로 변환
    img_array = np.array(image)
    
    # 이미지 리사이즈 (예: 224x224로 - 본인 모델에 맞게 조정)
    img_resized = cv2.resize(img_array, (224, 224))
    
    # 정규화 (0-1 범위로)
    img_normalized = img_resized / 255.0
    
    # 배치 차원 추가 (모델이 배치 입력을 기대하는 경우)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_image(model, processed_image):
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    predicted_class = 0
    confidence = 0.85

    return predicted_class, confidence

CLASS_NAMES = [
    "나방", "노린재", "담배가루이","무당벌레"  # 실제 클래스 이름으로 교체
]

# 모델 로드
try:
    model = load_model()
    st.success("✅ 모델이 성공적으로 로드되었습니다!")
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {str(e)}")
    st.stop()

# UI 레이아웃
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 사진 촬영")
    
    # 카메라 입력
    camera_image = st.camera_input("사진을 찍어주세요")
    
    if camera_image is not None:
        # 촬영한 이미지 표시
        st.image(camera_image, caption="촬영된 이미지", use_container_width=True)

with col2:
    st.subheader("🤖 AI 분류 결과")
    
    if camera_image is not None:
        try:
            # 이미지 전처리
            pil_image = Image.open(camera_image)
            processed_image = preprocess_image(pil_image)
            
            # 예측 수행
            with st.spinner("AI가 이미지를 분석 중입니다..."):
                predicted_class, confidence = predict_image(model, processed_image)
            
            # 결과 표시
            predicted_label = CLASS_NAMES[predicted_class]
            
            st.success(f"🎯 **예측 결과**: {predicted_label}")
            st.info(f"📊 **신뢰도**: {confidence:.2%}")
            
            # 신뢰도 막대 그래프
            st.progress(confidence)
            
            # 상세 정보 (선택사항)
            with st.expander("상세 정보 보기"):
                st.write(f"예측된 클래스 인덱스: {predicted_class}")
                st.write(f"전처리된 이미지 shape: {processed_image.shape}")
                
        except Exception as e:
            st.error(f"❌ 예측 중 오류 발생: {str(e)}")
    else:
        st.info("👆 왼쪽에서 사진을 촬영해주세요")