import streamlit as st
from PIL import Image
import tensorflow as tf

# Define class names globally
class_names= ['Bacterial Leaf Blight',
 'Brown Spot',
 'Healthy Rice Leaf',
 'Leaf Blast',
 'Leaf scald',
 'Sheath Blight']


# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize to model input size
    img = tf.convert_to_tensor(img)  # Convert to TensorFlow tensor
    img = tf.cast(img, dtype=tf.float32)  # Convert to float32
    img = img / 255.0  # Normalize pixel values
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to make predictions using your loaded model
def make_prediction(model, image):
    predictions = model.predict(image)  # Pass the preprocessed image to your model
    sorted_indices = tf.argsort(predictions[0])[::-1]  # Sort indices of predictions in descending order
    predicted_class = class_names[sorted_indices[0]]
    second_highest_class = class_names[sorted_indices[1]]
    return predicted_class, second_highest_class, predictions[0], sorted_indices

def get_disease_info (disease_name) :
    disease_info = {
        "Bacterial Leaf Blight" : "โรคขอบใบแห้ง วิธีป้องกันกำจัด \n\n1.ไม่นำเมล็ดพันธุ์จากแปลงเป็นโรคมาใช้ปลูก \n\n2.ใช้พันธุ์ข้าวที่ต้านทาน เช่น พันธุ์สุพรรณบุรี 60 สุพรรณบุรี 90 สุพรรณบุรี 1 สุพรรณบุรี 2 กข7 และ กข23 \n\n3.ในดินที่อุดมสมบูรณ์ไม่ควรใส่ปุ๋ยไนโตรเจนมาก",
        "Brown Spot" : "โรคใบจุดสีน้ำตาล วิธีป้องกันกำจัด \n\n1.ปรับปรุงดินโดยการไถกลบฟาง หรือเพิ่มความอุดมสมบูรณ์ดินโดยการปลูกพืชปุ๋ยสด หรือปลูกพืชหมุนเวียนเพื่อช่วยลดความรุนแรงของโรค \n\n2.คลุกเมล็ดพันธุ์ก่อนปลูกด้วยสารป้องกันกำจัดเชื้อรา เช่น แมนโคเซบ หรือคาร์เบนดาซิม+แมนโคเซบ อัตรา 3 กรัม/เมล็ด 1 กิโลกรัม \n\n3.ใส่ปุ๋ยโปแตสเซียมคลอไรด์ (0-0-60) อัตรา 5-10 กิโลกรัม/ไร่ ช่วยลดความรุนแรงของโรค \n\n4.ถ้าพบอาการของโรคใบจุดสีน้ำตาลรุนแรงทั่วไป 10 เปอร์เซ็นต์ของพื้นที่ใบในระยะข้าวแตกกอ หรือในระยะที่ต้นข้าวตั้งท้องใกล้ออกรวง เมื่อพบอาการใบจุดสีน้ำตาลที่ใบธงในสภาพฝนตกต่อเนื่อง อาจทำให้เกิดโรคเมล็ดด่าง ควรพ่นด้วยสารป้องกันกำจัดเชื้อรา เช่น อีดิเฟนฟอส คาร์เบนดาซิม แมนโคเซบ หรือ คาร์เบนดาซิม+แมนโคเซบ ตามอัตราที่ระบุ",
        "Healthy Rice Leaf" : "ใบข้าวแข็งแรงปกติดี",
        "Leaf Blast" : "โรคไหม้ วิธีป้องกันกำจัด \n\n1.เกษตรกรไม่ควรตกกล้าหรือหว่านข้าวหนาแน่นเกินไป อัตราที่เหมาะสมคือ 15 กก./ไร่  ในแปลงกล้าควรแบ่งแปลงย่อยให้มีพื้นที่พอเหมาะที่จะเข้าไปทำงานได้อย่างทั่วถึงและมีการถ่ายเทอากาศได้ดี \n\n2.หมั่นตรวจดูแปลงเป็นประจำ  โดยเฉพาะแปลงที่มีประวัติการระบาดมาก่อน  ถ้าเกษตรกรพบโรคไหม้ในระยะแรกจำนวนไม่มากสามารถกำจัดโดยตัดใบหรือถอนต้นเป็นโรคออกจากแปลง \n\n3.การคลุกเมล็ดพันธุ์ด้วยสารป้องกันกำจัดเชื้อราก่อนนำไปเพาะปลูก เช่น คาซูกามัยซิน ไตรไซคลาโซล คาร์เบนดาซิม  โพรคลอราซ  ตามอัตราที่แนะนำ \n\n4.ถ้าเกษตรกรพบโรคไหม้ระบาด ให้ทำการฉีดพ่นสารกำจัดเชื้อรา เช่น คาซูกามัยซิน คาร์เบนดาซิม อีดิเฟนฟอส  ไตรไซคลาโซล  ไอโซโพรไทโอเลน ตามอัตราที่แนะนำ ",
        "Leaf scald" : "โรคใบวงสีน้ำตาล วิธีป้องกันกำจัด \n\n1.ใช้พันธุ์ข้าวต้านทาน เช่น ในภาคตะวันออกเฉียงเหนือใช้ หางยี 71 \n\n2.กำจัดพืชอาศัยของเชื้อราสาเหตุโรค \n\n3.ในแหล่งที่เคยมีโรคระบาด หรือพบแผลลักษณะอาการดังที่กล่าวข้างต้นบนใบข้าวจำนวนมาก ในระยะข้าวแตกกอ ควรฉีดพ่นสารป้องกันกำจัดโรคพืช เช่น ไธโอฟาเนทเมทิล  โพรพิโคนาโซล ตามอัตราที่ระบุ",
        "Sheath Blight" : "โรคกาบใบแห้ง วิธีป้องกันกำจัด \n\n1.หลังเก็บเกี่ยวข้าว และเริ่มฤดูใหม่ ควรพลิกไถหน้าดิน เพื่อทำลายส่วนขยายพันธุ์ของเชื้อราสาเหตุโรค \n\n2.กำจัดวัชพืชตามคันนาและแหล่งน้ำ เพื่อลดโอกาสการพักตัวและเป็นแหล่งสะสมของเชื้อราสาเหตุโรค \n\n3.ใช้ชีวภัณฑ์ เช่น เชื้อแบคทีเรีย บาซิลลัส ซับทิลิส (เชื้อแบคทีเรียควบคุมเชื้อสาเหตุโรคพืช) ตามอัตราที่ระบุ \n\n4.ใช้สารป้องกันกำจัดเชื้อรา เช่น วาลิดามัยซิน โพรพิโคนาโซล เพนไซคูรอน หรืออีดิเฟนฟอส ตามอัตราที่ระบุโดยพ่นสารป้องกันกำจัดเชื้อรานี้ในบริเวณที่เริ่มพบโรคระบาด ไม่จำเป็นต้องพ่นทั้งแปลง เพราะโรคกาบใบแห้งจะเกิดเป็นหย่อม"
    }
    return disease_info.get(disease_name, "ไม่พบข้อมูล")


def main():
    st.title('ตรวจโรคใบข้าวด้วย Machine Learning (Rice Leaf Disease Predict)')
    st.info('This is a Rice Leaf Disease Predict using AI (CNN)')
    st.write("อัพโหลดรูปใบข้าว")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        - โรคขอบใบแห้ง (Bacterial Leaf Blight)
        - โรคใบจุดสีน้ำตาล (Brown Spot)
        - โรคไหม้ (Leaf Blast)
        """)
    with col2:
        st.write("""
        - โรคใบวงสีน้ำตาล (Leaf Scald)
        - โรคกาบใบแห้ง (Sheath Blight)
        - ใบข้าวสุขภาพดี (Healthy Rice Leaf)
        """)

    uploaded_file = st.file_uploader("เลือกอัพโหลดรูปใบข้าว", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            with st.spinner("กำลังประมวลผลรูปและทำนาย"):
                # Preprocess the image
                image = preprocess_image(uploaded_file)

                model = tf.keras.models.load_model("rice_model_ff.h5")

                # Make prediction and display results
                predicted_class, second_highest_class, probabilities, sorted_indices = make_prediction(model, image)
                disease_info = get_disease_info(predicted_class)

                st.subheader(
                    f"รูปใบข้าวที่ปรากฎคือ   : {predicted_class} ({probabilities[sorted_indices[0]] * 100:.2f}%)")
                st.write(
                    f"มีความเป็นไปได้ที่จะเป็น  : {second_highest_class} ({probabilities[sorted_indices[1]] * 100:.2f}%)")
                st.image(uploaded_file)
                st.subheader("ข้อมูลการป้องกันและกำจัด : "+predicted_class)
                st.write(disease_info)



        except Exception as e:
            st.error(f"Error processing image: {e}")


main()
