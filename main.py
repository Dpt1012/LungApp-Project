import asyncio
import json
import os
import pickle
import tempfile
import urllib.parse

import flet as ft
import flet_charts as fc
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import AuthenticationError, NotFoundError, OpenAI, PermissionDeniedError, RateLimitError

try:
    import flet_audio_recorder as far
except Exception:
    far = None

WHISPER_MODEL = "whisper-large-v3"
VOICE_INPUT_PATH = os.path.join(tempfile.gettempdir(), "lungcare_voice_input.wav")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"))

# --- 1. TẢI MÔ HÌNH RANDOM FOREST ---
MODEL_PATH = os.path.join(APP_DIR, "rf_model.pkl")
SEGMENTATION_MODEL_PATH = os.path.join(APP_DIR, "fpn_efficientnet_b4_gpu_p100_optimized_best.pt")


def load_model():
    if not os.path.exists(MODEL_PATH):
        return "ERROR_NOT_FOUND"
    try:
        import sklearn  # noqa: F401  Đảm bảo scikit-learn đã được cài
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        return f"ERROR_LOAD: {str(e)}"


ML_MODEL = load_model()

# --- 2. SEGMENT U PHỔI QUA CT ---
def segment_lung_tumor_scan(scan_path):
    from lung_tumor_segmentation import segment_lung_tumor_file

    return segment_lung_tumor_file(scan_path, SEGMENTATION_MODEL_PATH)


# --- 3. CẤU HÌNH API GROQ ---
MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
API_KEY_ERRORS = (AuthenticationError, RateLimitError, PermissionDeniedError)
MODEL_ACCESS_ERRORS = (NotFoundError,)

# ---------- BILINGUAL UI (giao diện Việt/Anh) ----------
MAPPING_GUIDE = {
    "vi": {
        "YELLOW_FINGERS": "Ngón tay ố vàng", "ANXIETY": "Hay lo lắng",
        "PEER_PRESSURE": "Áp lực bạn bè", "CHRONIC DISEASE": "Bệnh mãn tính (Tiểu đường, tim mạch...)",
        "FATIGUE ": "Mệt mỏi kéo dài", "ALLERGY ": "Dị ứng",
        "WHEEZING": "Thở khò khè", "ALCOHOL CONSUMING": "Thường xuyên uống rượu bia",
        "COUGHING": "Ho khan, ho có đờm", "SWALLOWING DIFFICULTY": "Khó nuốt",
        "CHEST PAIN": "Đau tức ngực",
    },
    "en": {
        "YELLOW_FINGERS": "Yellow-stained fingers", "ANXIETY": "Frequent anxiety",
        "PEER_PRESSURE": "Peer pressure", "CHRONIC DISEASE": "Chronic disease (diabetes, heart...)",
        "FATIGUE ": "Prolonged fatigue", "ALLERGY ": "Allergies",
        "WHEEZING": "Wheezing", "ALCOHOL CONSUMING": "Frequent alcohol consumption",
        "COUGHING": "Dry or phlegmy cough", "SWALLOWING DIFFICULTY": "Difficulty swallowing",
        "CHEST PAIN": "Chest pain",
    },
}
REQUIRED_FEATURES = list(MAPPING_GUIDE["vi"].keys())

TR = {
    "vi": {
        "window_title": "AI LungCare - Bác sĩ Tư vấn",
        "tooltip_home": "Trang chủ",
        "tooltip_dark_on": "Chuyển sang Chế độ Tối",
        "tooltip_dark_off": "Chuyển sang Chế độ Sáng",
        "tooltip_lang": "Switch to English",
        "tooltip_image": "Segment u phổi",

        "gate_heading": "Cần cấu hình Groq API Key",
        "gate_desc": "Ứng dụng cần một Groq API Key hợp lệ để Bác sĩ AI hoạt động. Lấy key miễn phí tại "
                     "console.groq.com/keys.",
        "gate_error": "API Key hiện tại đã hết hạn mức sử dụng hoặc không hợp lệ. Vui lòng nhập Key mới bên dưới.",
        "gate_confirm_btn": "Xác nhận",

        "sidebar_title": "Hồ sơ Y tế",
        "sidebar_progress": "Tiến độ: {n}/{total}",
        "sidebar_symptom_details": "Chi tiết triệu chứng:",
        "sidebar_yes": "Có",
        "sidebar_no": "Không",
        "sidebar_empty": "Trống",
        "sidebar_notes_title": "Ghi chú của bệnh nhân:",
        "sidebar_reset_btn": "Khám lại từ đầu",
        "sidebar_image_btn": "Segment u phổi",
        "sidebar_apikey_title": "Groq API Key",
        "sidebar_apikey_error": "Key hiện tại đã hết hạn mức hoặc không hợp lệ. Nhập Key mới bên dưới.",
        "sidebar_apikey_configured": "Đã cấu hình",
        "sidebar_apikey_missing": "Chưa có Key",
        "sidebar_apikey_status": "Trạng thái: {status}",
        "sidebar_apikey_new_label": "Nhập API Key mới",
        "sidebar_apikey_update_btn": "Cập nhật Key",

        "home_hero_title": "Hệ thống AI Tầm soát Hô hấp",
        "home_hero_desc": "Phòng khám thông minh sử dụng Mô hình Học máy (Machine Learning) kết hợp cùng Bác sĩ "
                          "Trợ lý Ảo để đánh giá nguy cơ các bệnh lý về phổi của bạn một cách nhanh chóng và bảo mật.",
        "home_cta_btn": "BẮT ĐẦU KHÁM TRỰC TUYẾN NGAY",
        "home_image_btn": "SEGMENT U PHỔI",
        "home_card1_title": "Giao tiếp Tự nhiên",
        "home_card1_desc": "Khai báo bệnh án thông qua trò chuyện tin nhắn, giọng nói hoặc trắc nghiệm trực quan.",
        "home_card2_title": "Tư duy Chuyên gia",
        "home_card2_desc": "Bác sĩ AI thông minh thấu hiểu ngữ cảnh, động viên và dẫn dắt người bệnh đi đúng trọng tâm.",
        "home_card3_title": "Báo cáo Chuyên sâu",
        "home_card3_desc": "Đánh giá tỷ lệ phần trăm rủi ro, phân tích biểu đồ sức khỏe và gợi ý bản đồ bệnh viện "
                           "chuyên khoa.",

        "chat_header_name": "Bác sĩ AI LungCare",
        "chat_header_status": "Đang trực tuyến • Sẵn sàng hỗ trợ bạn",
        "chat_input_hint": "Hoặc nhập nội dung trả lời, hỏi han Bác sĩ vào đây...",
        "chat_quick_yes": "Bị {symptom}",
        "chat_quick_no": "Không bị",
        "chat_mic_stop": "Dừng ghi âm",
        "chat_mic_start": "Ghi âm câu trả lời bằng giọng nói",
        "chat_mic_unavailable": "Micro không khả dụng trên môi trường này",
        "chat_thinking": "Bác sĩ AI đang suy nghĩ...",
        "chat_transcribing": "Đang chuyển giọng nói thành văn bản...",
        "chat_apikey_error_input": "⚠️ API Key đã hết hạn mức hoặc không hợp lệ. Vui lòng cập nhật Key mới ở "
                                    "thanh bên trái.",
        "chat_voice_error": "⚠️ Không nhận diện được giọng nói: {err}",
        "chat_mic_error": "⚠️ Không thể truy cập micro: {err}",
        "chat_apikey_error_send": "⚠️ API Key đã hết hạn mức hoặc không hợp lệ. Vui lòng cập nhật Key mới ở "
                                   "thanh bên trái rồi gửi lại tin nhắn.",
        "chat_ai_error": "⚠️ Lỗi hệ thống AI: {err}",
        "chat_model_error": "⚠️ Model AI hiện tại không khả dụng trên Groq: {model}. Vui lòng kiểm tra "
                            "GROQ_MODEL_NAME hoặc dùng openai/gpt-oss-120b.",
        "quick_reply_yes_echo": "Tôi có bị",
        "quick_reply_no_echo": "Tôi không bị",
        "quick_reply_saved_next": "Tôi đã lưu thông tin.\n\n👉 Tiếp theo, bạn có biểu hiện **{symptom}** không?",
        "quick_reply_complete": "✅ **Hồ sơ bệnh án đã hoàn tất!** Hệ thống đang tiến hành phân tích kết quả...",
        "advice_apikey_error": "⚠️ Không thể tạo lời khuyên vì API Key đã hết hạn mức. Vui lòng cập nhật Key mới ở "
                                "thanh bên trái, sau đó bấm 'Khám lại từ đầu'.",
        "advice_model_error": "⚠️ Không thể tạo lời khuyên vì model Groq hiện tại không khả dụng.",
        "advice_fallback": "Hệ thống khuyến cáo bạn duy trì lối sống lành mạnh, tập thể dục và thăm khám định kỳ.",

        "image_title": "Segment u phổi qua CT",
        "image_subtitle": "Tải file .npy hoặc ảnh CT slice để FPN EfficientNet-B4 tạo mask vùng u nghi ngờ. Model chạy trực tiếp "
                          "trên máy, không gửi ảnh lên Groq.",
        "image_change_btn": "Chọn file khác",
        "image_empty_title": "Chưa có CT/ảnh được chọn",
        "image_empty_desc": "Chọn CT volume .nii/.nii.gz hoặc ảnh CT slice để bắt đầu segment.",
        "image_file_label": "File đã chọn: {name}",
        "segmentation_pick_btn": "Segment u phổi (CT/NIfTI)",
        "segmentation_picker_title": "Chọn CT volume hoặc ảnh CT slice",
        "segmentation_supported": "FPN EfficientNet-B4 hỗ trợ tốt nhất các slice .npy từ dataset train của bạn "
                                  "theo preprocessing notebook: resize 256 và normalize [-1,1]. PNG/JPG dùng để demo.",
        "segmentation_loading": "Đang segment u phổi...",
        "segmentation_error_title": "Không thể segment ảnh/CT",
        "segmentation_model_missing": "Không tìm thấy model segment tại {path}.",
        "segmentation_result_title": "Kết quả segmentation FPN-B4",
        "segmentation_no_tumor": "Model chưa tạo vùng tumor trên dữ liệu này.",
        "segmentation_has_tumor": "Model đã tạo vùng tumor nghi ngờ.",
        "segmentation_area": "Tỷ lệ vùng tumor theo mask: {percent}%",
        "segmentation_probability": "Xác suất pixel tumor cao nhất: {prob}%",
        "segmentation_slices": "Số lát xử lý: {count}",
        "segmentation_preview": "Preview slice: {index}",
        "segmentation_mask_saved": "Mask lưu tại: {path}",
        "segmentation_prediction_saved": "Predicted mask lưu tại: {path}",
        "segmentation_visual_title": "Image và Predicted Mask",
        "segmentation_metric_area": "Tỷ lệ mask",
        "segmentation_metric_probability": "Max prob",
        "segmentation_metric_slices": "Số lát",
        "segmentation_metric_preview": "Slice xem",
        "segmentation_diagnosis_title": "Chẩn đoán AI từ kết quả segment",
        "segmentation_diagnosis_none": "AI chưa thấy vùng mask u rõ trên ảnh này. Nếu vẫn có triệu chứng hoặc bác sĩ "
                                       "nghi ngờ, bạn vẫn cần đọc phim/chụp kiểm tra theo chỉ định.",
        "segmentation_diagnosis_low": "AI phát hiện vùng u nghi ngờ rất nhỏ. Hãy xem đây là tín hiệu hỗ trợ và mang "
                                      "ảnh gốc cho bác sĩ hoặc bác sĩ chẩn đoán hình ảnh đánh giá.",
        "segmentation_diagnosis_suspicious": "AI phát hiện vùng u nghi ngờ trên CT. Bạn nên khám chuyên khoa hô hấp "
                                             "hoặc ung bướu và để bác sĩ đọc phim xác nhận.",
        "segmentation_diagnosis_high": "AI phát hiện vùng mask u nổi bật hơn trên ảnh. Cần được bác sĩ chuyên khoa "
                                       "đọc phim và chỉ định xét nghiệm/chẩn đoán tiếp theo.",
        "segmentation_diagnosis_disclaimer": "Đây là chẩn đoán AI hỗ trợ, không phải chẩn đoán y khoa cuối cùng. "
                                             "Kết luận chính thức phải do bác sĩ và hệ thống chẩn đoán lâm sàng thực hiện.",
        "segmentation_warning": "Lưu ý: model này dùng weight bạn train, best val Dice khoảng 0.823 trên notebook. "
                                "Kết quả chỉ hỗ trợ tham khảo, không thay thế chẩn đoán bác sĩ.",

        "result_error_title": "HỆ THỐNG GẶP LỖI KHI ĐỌC MÔ HÌNH NHÂN TẠO!",
        "result_error_cause": "Nguyên nhân: có thể do máy chưa cài đặt đúng scikit-learn.",
        "result_error_detail": "Chi tiết lỗi: {detail}",
        "result_error_path": "MODEL_PATH: {path}",
        "result_danger_title": "MỨC ĐỘ NGUY HIỂM",
        "result_danger_desc": "Phát hiện nhiều dấu hiệu lâm sàng nghiêm trọng. Cần đến ngay bệnh viện để chụp "
                               "X-Quang/CT phổi!",
        "result_warning_title": "MỨC ĐỘ CẢNH BÁO",
        "result_warning_desc": "Có dấu hiệu bệnh lý đường hô hấp. Khuyến nghị bạn sắp xếp đi thăm khám sớm.",
        "result_safe_title": "CHỈ SỐ AN TOÀN",
        "result_safe_desc": "Phổi của bạn có vẻ đang trong trạng thái khỏe mạnh. Hãy tiếp tục duy trì thói quen tốt nhé.",
        "result_gauge_title": "Đồng hồ Đánh giá AI",
        "result_radar_title": "Phân bổ Lâm sàng (Radar)",
        "result_analysis_title": "Phân tích Kết quả Tầm soát",
        "result_tab_advice": "Phác đồ Tư vấn Chuyên môn",
        "result_tab_hospitals": "Mạng lưới Bệnh viện Đề xuất",
        "result_advice_generating": "Đang tạo lời khuyên...",
        "result_advice_note_title": "Ghi chú từ Bác sĩ AI:",
        "result_hospitals_intro": "Dựa vào vị trí địa lý, hệ thống đề xuất các cơ sở Y tế uy tín sau:",
        "result_map_btn": "Mở hướng dẫn Google Maps",

        "welcome_message": (
            "Xin chào bạn! Tôi là Bác sĩ AI chuyên khoa Hô hấp của phòng khám LungCare, rất vui được đồng "
            "hành cùng bạn hôm nay. 😊\n\nTrước khi bắt đầu, bạn chia sẻ một chút nhé — dạo này sức khỏe bạn "
            "thế nào, có điều gì đang khiến bạn lo lắng hay muốn được tư vấn không?"
        ),
        "radar_categories": ["Ho đờm", "Đau ngực", "Khó thở/Khò khè", "Mệt mỏi", "Dị ứng", "Khó nuốt"],
    },
    "en": {
        "window_title": "AI LungCare - Doctor Consultation",
        "tooltip_home": "Home",
        "tooltip_dark_on": "Switch to Dark Mode",
        "tooltip_dark_off": "Switch to Light Mode",
        "tooltip_lang": "Chuyển sang tiếng Việt",
        "tooltip_image": "Segment lung tumor",

        "gate_heading": "Groq API Key Required",
        "gate_desc": "The app needs a valid Groq API Key for the AI Doctor to work. Get a free key at "
                     "console.groq.com/keys.",
        "gate_error": "The current API Key has run out of quota or is invalid. Please enter a new key below.",
        "gate_confirm_btn": "Confirm",

        "sidebar_title": "Medical Profile",
        "sidebar_progress": "Progress: {n}/{total}",
        "sidebar_symptom_details": "Symptom details:",
        "sidebar_yes": "Yes",
        "sidebar_no": "No",
        "sidebar_empty": "Empty",
        "sidebar_notes_title": "Patient notes:",
        "sidebar_reset_btn": "Start over",
        "sidebar_image_btn": "Segment lung tumor",
        "sidebar_apikey_title": "Groq API Key",
        "sidebar_apikey_error": "The current key has run out of quota or is invalid. Enter a new one below.",
        "sidebar_apikey_configured": "Configured",
        "sidebar_apikey_missing": "No key yet",
        "sidebar_apikey_status": "Status: {status}",
        "sidebar_apikey_new_label": "Enter new API Key",
        "sidebar_apikey_update_btn": "Update Key",

        "home_hero_title": "AI Respiratory Screening System",
        "home_hero_desc": "A smart clinic that combines Machine Learning with a Virtual Assistant Doctor to "
                          "quickly and securely assess your risk of lung disease.",
        "home_cta_btn": "START ONLINE CHECKUP NOW",
        "home_image_btn": "SEGMENT LUNG TUMOR",
        "home_card1_title": "Natural Conversation",
        "home_card1_desc": "Report your medical history through chat, voice, or quick multiple-choice replies.",
        "home_card2_title": "Expert Reasoning",
        "home_card2_desc": "A smart AI doctor that understands context, comforts you, and guides the conversation "
                           "to the point.",
        "home_card3_title": "In-depth Reports",
        "home_card3_desc": "Risk percentage assessment, health chart analysis, and specialist hospital "
                           "recommendations.",

        "chat_header_name": "AI Doctor LungCare",
        "chat_header_status": "Online • Ready to help you",
        "chat_input_hint": "Or type your reply, or chat with the Doctor here...",
        "chat_quick_yes": "Have {symptom}",
        "chat_quick_no": "Don't have it",
        "chat_mic_stop": "Stop recording",
        "chat_mic_start": "Record your reply by voice",
        "chat_mic_unavailable": "Microphone is unavailable in this environment",
        "chat_thinking": "AI Doctor is thinking...",
        "chat_transcribing": "Transcribing your voice to text...",
        "chat_apikey_error_input": "⚠️ The API Key has run out of quota or is invalid. Please update the key in "
                                    "the left sidebar.",
        "chat_voice_error": "⚠️ Could not recognize the voice: {err}",
        "chat_mic_error": "⚠️ Could not access the microphone: {err}",
        "chat_apikey_error_send": "⚠️ The API Key has run out of quota or is invalid. Please update the key in "
                                   "the left sidebar and resend your message.",
        "chat_ai_error": "⚠️ AI system error: {err}",
        "chat_model_error": "⚠️ The current Groq model is unavailable: {model}. Please check GROQ_MODEL_NAME or "
                            "use openai/gpt-oss-120b.",
        "quick_reply_yes_echo": "I do have",
        "quick_reply_no_echo": "I don't have",
        "quick_reply_saved_next": "I've noted that down.\n\n👉 Next, do you have **{symptom}**?",
        "quick_reply_complete": "✅ **Your medical profile is complete!** The system is now analyzing your results...",
        "advice_apikey_error": "⚠️ Could not generate advice because the API Key has run out of quota. Please "
                                "update the key in the left sidebar, then click 'Start over'.",
        "advice_model_error": "⚠️ Could not generate advice because the current Groq model is unavailable.",
        "advice_fallback": "The system recommends maintaining a healthy lifestyle, exercising, and having "
                            "regular checkups.",

        "image_title": "CT Lung Tumor Segmentation",
        "image_subtitle": "Upload a .npy file or CT slice image so FPN EfficientNet-B4 can create a suspected tumor mask. "
                          "The model runs locally and does not send images to Groq.",
        "image_change_btn": "Choose another file",
        "image_empty_title": "No CT/image selected",
        "image_empty_desc": "Choose a .nii/.nii.gz CT volume or CT slice image to start segmentation.",
        "image_file_label": "Selected file: {name}",
        "segmentation_pick_btn": "Segment lung tumor (CT/NIfTI)",
        "segmentation_picker_title": "Choose a CT volume or CT slice image",
        "segmentation_supported": "FPN EfficientNet-B4 works best with .npy slices from your training dataset, "
                                  "using the notebook preprocessing: resize to 256 and normalize to [-1,1]. PNG/JPG are for demos.",
        "segmentation_loading": "Segmenting lung tumor...",
        "segmentation_error_title": "Could not segment image/CT",
        "segmentation_model_missing": "Segmentation model was not found at {path}.",
        "segmentation_result_title": "FPN-B4 Segmentation Result",
        "segmentation_no_tumor": "The model did not create a tumor region for this data.",
        "segmentation_has_tumor": "The model created a suspicious tumor region.",
        "segmentation_area": "Tumor mask area ratio: {percent}%",
        "segmentation_probability": "Highest tumor-pixel probability: {prob}%",
        "segmentation_slices": "Slices processed: {count}",
        "segmentation_preview": "Preview slice: {index}",
        "segmentation_mask_saved": "Mask saved at: {path}",
        "segmentation_prediction_saved": "Predicted mask saved at: {path}",
        "segmentation_visual_title": "Image and Predicted Mask",
        "segmentation_metric_area": "Mask area",
        "segmentation_metric_probability": "Max prob",
        "segmentation_metric_slices": "Slices",
        "segmentation_metric_preview": "Preview",
        "segmentation_diagnosis_title": "AI diagnosis from segmentation",
        "segmentation_diagnosis_none": "The AI did not find a clear tumor-mask region in this image. If symptoms "
                                       "persist or a clinician is concerned, you still need medical image review or "
                                       "follow-up imaging as directed.",
        "segmentation_diagnosis_low": "The AI found a very small suspicious region. Treat this as a support signal and "
                                      "bring the original scan to a doctor or radiologist for review.",
        "segmentation_diagnosis_suspicious": "The AI found a suspicious tumor region on the CT image. You should see a "
                                             "pulmonology or oncology specialist and have a doctor confirm the scan.",
        "segmentation_diagnosis_high": "The AI found a more prominent tumor-mask region in the image. A specialist "
                                       "should review the scan and decide the next diagnostic tests.",
        "segmentation_diagnosis_disclaimer": "This is an AI-assisted diagnosis, not a final medical diagnosis. The "
                                             "official conclusion must come from a doctor and clinical diagnostic workflow.",
        "segmentation_warning": "Note: this uses your trained weights, with best validation Dice around 0.823 in "
                                "the notebook. Results are only a support signal and do not replace diagnosis.",

        "result_error_title": "SYSTEM ERROR WHILE LOADING THE AI MODEL!",
        "result_error_cause": "Cause: the machine may not have scikit-learn installed correctly.",
        "result_error_detail": "Error detail: {detail}",
        "result_error_path": "MODEL_PATH: {path}",
        "result_danger_title": "HIGH RISK LEVEL",
        "result_danger_desc": "Several serious clinical signs detected. Please go to a hospital immediately for "
                               "a chest X-ray/CT scan!",
        "result_warning_title": "WARNING LEVEL",
        "result_warning_desc": "Signs of respiratory disease detected. We recommend arranging a checkup soon.",
        "result_safe_title": "SAFE LEVEL",
        "result_safe_desc": "Your lungs appear to be in a healthy state. Keep up your good habits.",
        "result_gauge_title": "AI Risk Gauge",
        "result_radar_title": "Clinical Distribution (Radar)",
        "result_analysis_title": "Screening Result Analysis",
        "result_tab_advice": "Specialist Advice",
        "result_tab_hospitals": "Recommended Hospital Network",
        "result_advice_generating": "Generating advice...",
        "result_advice_note_title": "Notes from the AI Doctor:",
        "result_hospitals_intro": "Based on your location, the system recommends these trusted medical facilities:",
        "result_map_btn": "Open Google Maps directions",

        "welcome_message": (
            "Hello! I'm the AI Respiratory Doctor at LungCare clinic, glad to be here with you today. 😊\n\n"
            "Before we start, tell me a little — how has your health been lately, is there anything worrying you "
            "or that you'd like advice on?"
        ),
        "radar_categories": ["Cough", "Chest pain", "Wheezing/SOB", "Fatigue", "Allergy", "Swallowing difficulty"],
    },
}

PRIMARY = "#0284c7"
RED = "#ef4444"
AMBER = "#d97706"
GREEN = "#059669"

FONT_FAMILY = "BeVietnamPro"
FONT_URL = "https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;900&display=swap"

SOFT_SHADOW = ft.BoxShadow(blur_radius=16, spread_radius=0, color=ft.Colors.with_opacity(0.10, "#000000"))


def main(page: ft.Page):
    page.padding = 0
    page.scroll = ft.ScrollMode.AUTO
    page.fonts = {FONT_FAMILY: FONT_URL}
    page.theme = ft.Theme(font_family=FONT_FAMILY, color_scheme_seed=PRIMARY, use_material3=True)
    page.dark_theme = ft.Theme(font_family=FONT_FAMILY, color_scheme_seed=PRIMARY, use_material3=True)
    page.theme_mode = ft.ThemeMode.LIGHT

    state = {
        "page": "home",
        "patient_data": {feat: None for feat in REQUIRED_FEATURES},
        "extra_symptoms": "",
        "messages": [],
        "risk_score": None,
        "advice_text": None,
        "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
        "api_key_error": False,
        "chat_status": "",
        "segmentation_path": None,
        "segmentation_result": None,
        "segmentation_error": None,
        "segmentation_loading": False,
        "result_tab": "advice",
        "dark_mode": False,
        "ui_language": "vi",
        "language": None,
        "recording": False,
        "ui": {},
    }

    def t(key):
        return TR[state["ui_language"]][key]

    def sym(code):
        return MAPPING_GUIDE[state["ui_language"]][code]

    def chat_lang():
        # The quick-reply area follows whatever language the patient is actually chatting in
        # (auto-detected), falling back to the manually-toggled UI language before that's known.
        return state["language"] if state["language"] in MAPPING_GUIDE else state["ui_language"]

    def chat_t(key):
        return TR[chat_lang()][key]

    def chat_sym(code):
        return MAPPING_GUIDE[chat_lang()][code]

    def welcome_message():
        return {"role": "assistant", "content": t("welcome_message")}

    state["messages"] = [welcome_message()]

    content_area = ft.Container(expand=True)
    audio_recorder = None
    if far is not None:
        try:
            audio_recorder = far.AudioRecorder(configuration=far.AudioRecorderConfiguration(encoder=far.AudioEncoder.WAV))
            page.services.append(audio_recorder)
        except Exception:
            audio_recorder = None
    image_file_picker = ft.FilePicker()
    page.services.append(image_file_picker)
    page.add(content_area)

    def get_client():
        return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=state["groq_api_key"])

    def groq_extra_body():
        if MODEL_NAME.startswith("openai/gpt-oss"):
            return {"include_reasoning": False, "reasoning_effort": "low"}
        if MODEL_NAME.startswith("qwen/"):
            return {"reasoning_format": "hidden", "reasoning_effort": "none"}
        return {}

    def save_new_key(new_key: str):
        state["groq_api_key"] = new_key.strip()
        state["api_key_error"] = False

    # ---------- TOP-LEVEL NAVIGATION (full page swap — rare, cheap enough) ----------
    def navigate_gate():
        content_area.content = build_api_key_gate()
        content_area.update()

    def navigate_home(e=None):
        state["page"] = "home"
        content_area.content = build_home()
        content_area.update()

    def ensure_shell():
        if "shell" not in state["ui"]:
            sidebar_ctrl = ft.Container(content=build_sidebar_content(), width=280, padding=20)
            main_area_ctrl = ft.Container(expand=True, padding=25)
            state["ui"]["sidebar_ctrl"] = sidebar_ctrl
            state["ui"]["main_area_ctrl"] = main_area_ctrl
            state["ui"]["shell"] = ft.Row(
                [sidebar_ctrl, ft.VerticalDivider(width=1), main_area_ctrl], expand=True,
            )
        else:
            refresh_sidebar()

    def refresh_sidebar():
        ctrl = state["ui"].get("sidebar_ctrl")
        if ctrl is None:
            return
        ctrl.content = build_sidebar_content()
        if ctrl.page:
            ctrl.update()

    def navigate_chat(e=None):
        state["page"] = "chat"
        ensure_shell()
        build_chat_view()
        content_area.content = state["ui"]["shell"]
        content_area.update()

    def navigate_result():
        state["page"] = "result"
        ensure_shell()
        build_result_view()
        content_area.content = state["ui"]["shell"]
        content_area.update()

    def navigate_image(e=None):
        state["page"] = "image"
        ensure_shell()
        build_image_view()
        content_area.content = state["ui"]["shell"]
        content_area.update()

    def reset_app(e=None):
        keep_key = state["groq_api_key"]
        state.update({
            "page": "home",
            "patient_data": {feat: None for feat in REQUIRED_FEATURES},
            "extra_symptoms": "",
            "messages": [welcome_message()],
            "risk_score": None,
            "advice_text": None,
            "api_key_error": False,
            "chat_status": "",
            "segmentation_path": None,
            "segmentation_result": None,
            "segmentation_error": None,
            "segmentation_loading": False,
            "result_tab": "advice",
            "language": None,
            "recording": False,
            "ui": {},
        })
        state["groq_api_key"] = keep_key
        navigate_home()

    def refresh_current_page():
        if state["page"] == "chat":
            ensure_shell()
            build_chat_view()
            content_area.content = state["ui"]["shell"]
            content_area.update()
        elif state["page"] == "result":
            ensure_shell()
            state["ui"].pop("result_static", None)
            build_result_view()
            content_area.content = state["ui"]["shell"]
            content_area.update()
        elif state["page"] == "image":
            ensure_shell()
            build_image_view()
            content_area.content = state["ui"]["shell"]
            content_area.update()
        else:
            navigate_home()

    # ---------- API KEY ----------
    def build_api_key_gate():
        key_field = ft.TextField(label="Groq API Key", password=True, can_reveal_password=True, hint_text="gsk_...")

        def confirm(e):
            if key_field.value and key_field.value.strip():
                save_new_key(key_field.value)
                navigate_home()

        controls = [
            ft.CircleAvatar(content=ft.Icon(ft.Icons.VPN_KEY_ROUNDED, color="white", size=32), bgcolor=PRIMARY,
                             radius=36),
            ft.Text(t("gate_heading"), size=24, weight=ft.FontWeight.BOLD, color=PRIMARY,
                    text_align=ft.TextAlign.CENTER),
            ft.Text(t("gate_desc"), opacity=0.8, text_align=ft.TextAlign.CENTER),
        ]
        if state["api_key_error"]:
            controls.append(ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=RED, size=18),
                ft.Text(t("gate_error"), color=RED, expand=True),
            ]))
        controls += [
            key_field,
            ft.ElevatedButton(t("gate_confirm_btn"), icon=ft.Icons.CHECK_ROUNDED, on_click=confirm, bgcolor=PRIMARY,
                               color="white", width=480, height=46),
        ]

        return ft.Container(
            content=ft.Container(
                content=ft.Column(controls, spacing=15, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                   width=480),
                padding=40, border_radius=20, bgcolor=ft.Colors.with_opacity(0.04, PRIMARY),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.15, "#808080")), shadow=SOFT_SHADOW,
            ),
            padding=60, alignment=ft.Alignment.CENTER, expand=True,
        )

    def build_api_key_section():
        key_field = ft.TextField(label=t("sidebar_apikey_new_label"), password=True, can_reveal_password=True,
                                  hint_text="gsk_...")

        def update_key(e):
            if key_field.value and key_field.value.strip():
                save_new_key(key_field.value)
                refresh_sidebar()

        status_icon = ft.Icons.CHECK_CIRCLE_ROUNDED if state["groq_api_key"] else ft.Icons.ERROR_ROUNDED
        status_color = GREEN if state["groq_api_key"] else AMBER
        status_text = t("sidebar_apikey_configured") if state["groq_api_key"] else t("sidebar_apikey_missing")
        items = [
            ft.Divider(),
            ft.Row([ft.Icon(ft.Icons.VPN_KEY_ROUNDED, size=18, color=PRIMARY),
                    ft.Text(t("sidebar_apikey_title"), weight=ft.FontWeight.BOLD)], spacing=6),
        ]
        if state["api_key_error"]:
            items.append(ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=16, color=RED),
                ft.Text(t("sidebar_apikey_error"), color=RED, size=12, expand=True),
            ]))
        items += [
            ft.Row([ft.Icon(status_icon, size=16, color=status_color),
                    ft.Text(t("sidebar_apikey_status").format(status=status_text), size=12, opacity=0.7)],
                   spacing=6),
            key_field,
            ft.ElevatedButton(t("sidebar_apikey_update_btn"), icon=ft.Icons.SYNC_ROUNDED, on_click=update_key,
                               bgcolor=PRIMARY, color="white"),
        ]
        return ft.Column(items, spacing=8)

    # ---------- SIDEBAR ----------
    def build_sidebar_content():
        collected = [k for k, v in state["patient_data"].items() if v is not None]
        total = len(REQUIRED_FEATURES)
        rows = [
            ft.Row([ft.Icon(ft.Icons.FOLDER_SHARED_ROUNDED, color=PRIMARY, size=20),
                    ft.Text(t("sidebar_title"), size=18, weight=ft.FontWeight.BOLD, color=PRIMARY)],
                   alignment=ft.MainAxisAlignment.CENTER, spacing=6),
            ft.Text(t("sidebar_progress").format(n=len(collected), total=total), weight=ft.FontWeight.BOLD,
                    text_align=ft.TextAlign.CENTER),
            ft.ProgressBar(value=len(collected) / total, color=PRIMARY),
            ft.Divider(),
            ft.Row([ft.Icon(ft.Icons.ASSIGNMENT_ROUNDED, size=16, color=PRIMARY),
                    ft.Text(t("sidebar_symptom_details"), weight=ft.FontWeight.BOLD)], spacing=6),
        ]
        for k, v in state["patient_data"].items():
            label = sym(k).split("(")[0].strip()
            if v is not None:
                status = (ft.Row([ft.Icon(ft.Icons.CANCEL_ROUNDED, size=14, color=RED),
                                   ft.Text(t("sidebar_yes"), color=RED, weight=ft.FontWeight.BOLD)], spacing=4)
                          if v == 1
                          else ft.Row([ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED, size=14, color=GREEN),
                                        ft.Text(t("sidebar_no"), color=GREEN, weight=ft.FontWeight.BOLD)],
                                       spacing=4))
            else:
                status = ft.Row([
                    ft.Icon(ft.Icons.HOURGLASS_EMPTY_ROUNDED, size=14, opacity=0.4),
                    ft.Text(t("sidebar_empty"), size=13, opacity=0.4),
                ], spacing=4)

            rows.append(ft.Row([
                ft.Text(
                    label,
                    size=13,
                    opacity=0.4 if v is None else 1.0,
                    expand=True,
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.Container(
                    content=status,
                    width=64,
                    alignment=ft.Alignment.CENTER_RIGHT,
                ),
            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER))

        if state["extra_symptoms"]:
            rows.append(ft.Divider())
            rows.append(ft.Row([ft.Icon(ft.Icons.STICKY_NOTE_2_ROUNDED, size=16, color=PRIMARY),
                                 ft.Text(t("sidebar_notes_title"), weight=ft.FontWeight.BOLD)], spacing=6))
            rows.append(ft.Text(state["extra_symptoms"], color=PRIMARY, size=13))

        rows.append(ft.Divider())
        rows.append(ft.ElevatedButton(t("sidebar_image_btn"), icon=ft.Icons.IMAGE_SEARCH_ROUNDED,
                                       on_click=navigate_image, width=196))
        rows.append(ft.ElevatedButton(t("sidebar_reset_btn"), icon=ft.Icons.REFRESH_ROUNDED, on_click=reset_app,
                                       width=196))
        rows.append(build_api_key_section())

        return ft.Card(
            content=ft.Container(
                content=ft.Column(rows, spacing=8, scroll=ft.ScrollMode.AUTO), padding=18,
            ),
            elevation=1,
        )

    # ---------- HOME ----------
    def feature_card(icon, title, desc, accent):
        return ft.Card(elevation=2, content=ft.Container(
            content=ft.Column([
                ft.CircleAvatar(content=ft.Icon(icon, color="white", size=26), bgcolor=accent, radius=30),
                ft.Text(title, size=18, weight=ft.FontWeight.BOLD, color=PRIMARY),
                ft.Text(desc, opacity=0.8, text_align=ft.TextAlign.CENTER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
            padding=25,
        ))

    def build_home():
        hero_badge = ft.Stack([
            ft.Container(width=160, height=160, border_radius=160,
                         bgcolor=ft.Colors.with_opacity(0.12, PRIMARY)),
            ft.Container(
                content=ft.Icon(ft.Icons.HEALTH_AND_SAFETY_ROUNDED, color=PRIMARY, size=76),
                width=160, height=160, alignment=ft.Alignment.CENTER,
            ),
        ], width=160, height=160)

        hero = ft.Container(
            content=ft.Column([
                hero_badge,
                ft.Text(t("home_hero_title"), size=36, weight=ft.FontWeight.W_900, color=PRIMARY,
                        text_align=ft.TextAlign.CENTER),
                ft.Text(t("home_hero_desc"), size=16, opacity=0.8, text_align=ft.TextAlign.CENTER),
                ft.Row([
                    ft.ElevatedButton(t("home_cta_btn"), icon=ft.Icons.ARROW_FORWARD_ROUNDED,
                                      on_click=navigate_chat, bgcolor=PRIMARY, color="white", width=400, height=48),
                    ft.OutlinedButton(t("home_image_btn"), icon=ft.Icons.IMAGE_SEARCH_ROUNDED,
                                      on_click=navigate_image, width=260, height=48),
                ], alignment=ft.MainAxisAlignment.CENTER, wrap=True, spacing=12),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20),
            padding=60, alignment=ft.Alignment.CENTER, border_radius=24,
            bgcolor=ft.Colors.with_opacity(0.04, PRIMARY),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.15, "#808080")),
        )
        cards = ft.ResponsiveRow([
            ft.Container(feature_card(ft.Icons.CHAT_BUBBLE_ROUNDED, t("home_card1_title"), t("home_card1_desc"),
                                       PRIMARY), col=4),
            ft.Container(feature_card(ft.Icons.SMART_TOY_ROUNDED, t("home_card2_title"), t("home_card2_desc"),
                                       GREEN), col=4),
            ft.Container(feature_card(ft.Icons.INSIGHTS_ROUNDED, t("home_card3_title"), t("home_card3_desc"),
                                       AMBER), col=4),
        ])
        return ft.Container(
            content=ft.Column([hero, cards], spacing=25, scroll=ft.ScrollMode.AUTO, expand=True),
            padding=30, expand=True,
        )

    # ---------- CT SEGMENTATION ----------
    def segmentation_model_error_text():
        if not os.path.exists(SEGMENTATION_MODEL_PATH):
            return t("segmentation_model_missing").format(path=SEGMENTATION_MODEL_PATH)
        return ""

    def build_segmentation_result_card():
        model_error = segmentation_model_error_text()
        if model_error:
            return ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.ERROR_ROUNDED, color=RED, size=24),
                    ft.Text(model_error, color=RED, expand=True),
                ], spacing=10),
                bgcolor=ft.Colors.with_opacity(0.08, RED),
                border=ft.Border.all(1, RED),
                border_radius=14,
                padding=18,
            )

        if state["segmentation_loading"]:
            return ft.Container(
                content=ft.Row([
                    ft.ProgressRing(width=22, height=22, stroke_width=3, color=PRIMARY),
                    ft.Text(t("segmentation_loading"), italic=True, opacity=0.8),
                ], spacing=12),
                padding=18,
            )

        if state["segmentation_error"]:
            return ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=RED, size=24),
                        ft.Text(t("segmentation_error_title"), weight=ft.FontWeight.BOLD, color=RED),
                    ], spacing=8),
                    ft.Text(state["segmentation_error"], selectable=True, size=12, opacity=0.8),
                    ft.Text(t("segmentation_warning"), size=12, opacity=0.7),
                ], spacing=8),
                bgcolor=ft.Colors.with_opacity(0.08, RED),
                border=ft.Border.all(1, RED),
                border_radius=14,
                padding=18,
            )

        result = state["segmentation_result"]
        if not result:
            return ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.MY_LOCATION_ROUNDED, color=PRIMARY, size=20),
                        ft.Text(t("segmentation_supported"), expand=True, opacity=0.82),
                    ], spacing=8),
                    ft.Text(t("segmentation_warning"), size=12, opacity=0.65),
                ], spacing=10),
                bgcolor=ft.Colors.with_opacity(0.05, PRIMARY),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.25, PRIMARY)),
                border_radius=14,
                padding=18,
            )

        color = RED if result["has_tumor"] else GREEN
        icon = ft.Icons.TRACK_CHANGES_ROUNDED if result["has_tumor"] else ft.Icons.CHECK_CIRCLE_ROUNDED
        title = t("segmentation_has_tumor") if result["has_tumor"] else t("segmentation_no_tumor")

        def metric_tile(icon_name, label, value):
            return ft.Container(
                content=ft.Row([
                    ft.Icon(icon_name, color=color, size=18),
                    ft.Column([
                        ft.Text(label, size=11, opacity=0.62),
                        ft.Text(str(value), size=17, weight=ft.FontWeight.BOLD, color=color),
                    ], spacing=1),
                ], spacing=8),
                width=150,
                padding=12,
                border_radius=10,
                bgcolor=ft.Colors.with_opacity(0.06, color),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.22, color)),
            )

        def ai_diagnosis():
            tumor_percent = float(result.get("tumor_percent") or 0)
            max_probability = float(result.get("max_probability") or 0)

            if not result["has_tumor"] or result.get("tumor_voxels", 0) <= 0:
                return (
                    t("segmentation_diagnosis_none"),
                    GREEN,
                    ft.Icons.CHECK_CIRCLE_ROUNDED,
                )
            if tumor_percent >= 1.0 or (tumor_percent >= 0.5 and max_probability >= 90):
                return (
                    t("segmentation_diagnosis_high"),
                    RED,
                    ft.Icons.DANGEROUS_ROUNDED,
                )
            if tumor_percent >= 0.1 or max_probability >= 80:
                return (
                    t("segmentation_diagnosis_suspicious"),
                    AMBER,
                    ft.Icons.WARNING_ROUNDED,
                )
            return (
                t("segmentation_diagnosis_low"),
                AMBER,
                ft.Icons.INFO_ROUNDED,
            )

        diagnosis_text, diagnosis_color, diagnosis_icon = ai_diagnosis()

        content = [
            ft.Row([
                ft.Icon(icon, color=color, size=30),
                ft.Column([
                    ft.Text(t("segmentation_result_title"), size=18, weight=ft.FontWeight.BOLD, color=color),
                    ft.Text(title, size=13, opacity=0.78),
                ], spacing=2, expand=True),
            ], spacing=10),
        ]
        content.extend([
            ft.Row([
                metric_tile(ft.Icons.PIE_CHART_ROUNDED, t("segmentation_metric_area"),
                            f'{result["tumor_percent"]}%'),
                metric_tile(ft.Icons.SPEED_ROUNDED, t("segmentation_metric_probability"),
                            f'{result["max_probability"]}%'),
                metric_tile(ft.Icons.VIEW_STREAM_ROUNDED, t("segmentation_metric_slices"),
                            result["slices_processed"]),
                metric_tile(ft.Icons.IMAGE_ROUNDED, t("segmentation_metric_preview"),
                            result["preview_slice"]),
            ], wrap=True, spacing=10, run_spacing=10),
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(diagnosis_icon, color=diagnosis_color, size=20),
                        ft.Text(t("segmentation_diagnosis_title"), size=14, weight=ft.FontWeight.BOLD,
                                color=diagnosis_color, expand=True),
                    ], spacing=8),
                    ft.Text(diagnosis_text, size=13, opacity=0.86),
                    ft.Text(t("segmentation_diagnosis_disclaimer"), size=12, weight=ft.FontWeight.BOLD,
                            color=diagnosis_color),
                ], spacing=8),
                padding=14,
                border_radius=12,
                bgcolor=ft.Colors.with_opacity(0.08, diagnosis_color),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.35, diagnosis_color)),
            ),
            ft.Text(t("segmentation_prediction_saved").format(path=result.get("prediction_mask_path", "")),
                    selectable=True, size=12, opacity=0.75),
            ft.Text(t("segmentation_mask_saved").format(path=result["mask_path"]),
                    selectable=True, size=12, opacity=0.75),
            ft.Text(t("segmentation_warning"), size=12, opacity=0.65),
        ])

        return ft.Container(
            content=ft.Column(content, spacing=10),
            bgcolor=ft.Colors.with_opacity(0.08, color),
            border=ft.Border.all(1, color),
            border_radius=14,
            padding=20,
        )

    async def pick_segmentation_scan(e):
        model_error = segmentation_model_error_text()
        if model_error:
            state["segmentation_error"] = model_error
            refresh_image_view()
            return

        files = await image_file_picker.pick_files(
            dialog_title=t("segmentation_picker_title"),
            file_type=ft.FilePickerFileType.ANY,
            allowed_extensions=["nii", "gz", "npy", "png", "jpg", "jpeg", "bmp"],
            allow_multiple=False,
        )
        if not files:
            return

        selected_path = getattr(files[0], "path", None)
        if not selected_path:
            return

        state["segmentation_path"] = selected_path
        state["segmentation_result"] = None
        state["segmentation_error"] = None
        state["segmentation_loading"] = True
        refresh_image_view()
        try:
            state["segmentation_result"] = await asyncio.to_thread(segment_lung_tumor_scan, selected_path)
        except Exception as ex:
            state["segmentation_error"] = str(ex)
        finally:
            state["segmentation_loading"] = False
            refresh_image_view()

    def build_segmentation_preview():
        def visual_panel(content, label_icon, label_text):
            return ft.Column([
                ft.Row([
                    ft.Container(
                        content=content,
                        height=470,
                        expand=True,
                        padding=12,
                        alignment=ft.Alignment.CENTER,
                        bgcolor=ft.Colors.with_opacity(0.04, PRIMARY),
                        border=ft.Border.all(1, ft.Colors.with_opacity(0.18, "#808080")),
                        border_radius=14,
                    ),
                ]),
                ft.Row([
                    ft.Icon(label_icon, size=16, color=PRIMARY),
                    ft.Text(label_text, size=12, opacity=0.75, expand=True, selectable=True),
                ], spacing=6),
            ], spacing=8)

        result = state["segmentation_result"]
        comparison_path = result.get("comparison_path") if result else None
        if comparison_path and os.path.exists(comparison_path):
            return visual_panel(
                ft.Image(
                    src=comparison_path,
                    width=760,
                    height=430,
                    fit=ft.BoxFit.CONTAIN,
                    border_radius=10,
                ),
                ft.Icons.TRACK_CHANGES_ROUNDED,
                t("segmentation_visual_title"),
            )

        if state["segmentation_path"]:
            lower_name = os.path.basename(state["segmentation_path"]).lower()
            can_preview = lower_name.endswith((".png", ".jpg", ".jpeg", ".bmp"))
            preview_content = (
                ft.Image(
                    src=state["segmentation_path"],
                    width=760,
                    height=430,
                    fit=ft.BoxFit.CONTAIN,
                    border_radius=10,
                )
                if can_preview
                else ft.Column([
                    ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, size=70, color=PRIMARY),
                    ft.Text(os.path.basename(state["segmentation_path"]),
                            weight=ft.FontWeight.BOLD, color=PRIMARY, text_align=ft.TextAlign.CENTER),
                    ft.Text(t("segmentation_supported"), size=12, opacity=0.7, text_align=ft.TextAlign.CENTER),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                   alignment=ft.MainAxisAlignment.CENTER, spacing=8)
            )
            return visual_panel(
                preview_content,
                ft.Icons.FOLDER_OPEN_ROUNDED,
                t("image_file_label").format(name=os.path.basename(state["segmentation_path"])),
            )

        return visual_panel(
            ft.Column([
                ft.Icon(ft.Icons.TRACK_CHANGES_ROUNDED, size=70, color=PRIMARY),
                ft.Text(t("image_empty_title"), size=18, weight=ft.FontWeight.BOLD, color=PRIMARY),
                ft.Text(t("image_empty_desc"), opacity=0.7, text_align=ft.TextAlign.CENTER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER,
               alignment=ft.MainAxisAlignment.CENTER, spacing=8),
            ft.Icons.INFO_ROUNDED,
            t("segmentation_supported"),
        )

    def build_image_content():
        button_label = t("image_change_btn") if state["segmentation_path"] else t("segmentation_pick_btn")
        pick_button = ft.ElevatedButton(
            button_label,
            icon=ft.Icons.TRACK_CHANGES_ROUNDED,
            on_click=pick_segmentation_scan,
            bgcolor=PRIMARY,
            color="white",
            height=44,
            disabled=bool(segmentation_model_error_text()),
        )

        header = ft.Card(elevation=1, content=ft.Container(
            content=ft.Row([
                ft.CircleAvatar(content=ft.Icon(ft.Icons.BIOTECH_ROUNDED, color="white", size=26),
                                 bgcolor=PRIMARY, radius=26),
                ft.Column([
                    ft.Text(t("image_title"), size=20, weight=ft.FontWeight.BOLD, color=PRIMARY),
                    ft.Text(t("image_subtitle"), size=13, opacity=0.72),
                ], spacing=3, expand=True),
                pick_button,
            ], spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=15,
        ))

        body = ft.Column([
            ft.Card(elevation=2, content=ft.Container(
                content=build_segmentation_preview(),
                padding=18,
                alignment=ft.Alignment.CENTER,
            )),
            ft.Card(elevation=2, content=ft.Container(
                content=build_segmentation_result_card(),
                padding=18,
            )),
        ], spacing=12)

        return ft.Column([header, body], spacing=15, scroll=ft.ScrollMode.AUTO, expand=True)

    def refresh_image_view():
        if state["page"] != "image":
            return
        ctrl = state["ui"].get("main_area_ctrl")
        if ctrl is None:
            return
        ctrl.content = build_image_content()
        if ctrl.page:
            ctrl.update()

    def build_image_view():
        state["ui"]["main_area_ctrl"].content = build_image_content()

    # ---------- CHAT ----------
    def message_bubble(msg):
        is_user = msg["role"] == "user"
        avatar = ft.CircleAvatar(
            content=ft.Icon(ft.Icons.PERSON_ROUNDED if is_user else ft.Icons.MEDICAL_SERVICES_ROUNDED,
                             color="white", size=18),
            bgcolor="#6b7280" if is_user else PRIMARY, radius=18,
        )
        bubble = ft.Container(
            content=ft.Markdown(msg["content"], selectable=True),
            bgcolor=ft.Colors.with_opacity(0.12, PRIMARY) if not is_user else ft.Colors.with_opacity(0.08, "#6b7280"),
            padding=14, border_radius=14, width=560,
        )
        row_controls = [bubble, avatar] if is_user else [avatar, bubble]
        return ft.Row(row_controls, alignment=ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START)

    def add_chat_bubble(role, content):
        state["messages"].append({"role": role, "content": content})
        chat_list = state["ui"].get("chat_list_ctrl")
        if chat_list is not None:
            chat_list.controls.append(message_bubble(state["messages"][-1]))
            chat_list.update()

    async def toggle_recording(e):
        if audio_recorder is None:
            add_chat_bubble("assistant", chat_t("chat_mic_unavailable"))
            refresh_chat_bottom()
            return

        if state["recording"]:
            state["recording"] = False
            refresh_chat_bottom()
            path = await audio_recorder.stop_recording()
            if not path or not os.path.exists(path):
                return
            state["chat_status"] = t("chat_transcribing")
            refresh_chat_bottom()
            try:
                with open(path, "rb") as f:
                    # Whisper's API only accepts a single language hint (no shortlist support), so we
                    # bias it toward whichever of the app's two supported languages (vi/en) the user has
                    # explicitly chosen via the UI toggle — an intentional, user-controlled hint, unlike
                    # the old bug where a stale auto-detected chat language silently forced the wrong one.
                    transcript = await asyncio.to_thread(
                        get_client().audio.transcriptions.create, model=WHISPER_MODEL, file=f,
                        language=state["ui_language"],
                    )
                state["chat_status"] = ""
                await handle_send(transcript.text)
            except API_KEY_ERRORS:
                state["api_key_error"] = True
                state["chat_status"] = ""
                add_chat_bubble("assistant", t("chat_apikey_error_input"))
                refresh_chat_bottom()
            except Exception as ex:
                state["chat_status"] = ""
                add_chat_bubble("assistant", t("chat_voice_error").format(err=ex))
                refresh_chat_bottom()
        else:
            try:
                await audio_recorder.start_recording(VOICE_INPUT_PATH)
                state["recording"] = True
                refresh_chat_bottom()
            except Exception as ex:
                add_chat_bubble("assistant", t("chat_mic_error").format(err=ex))
                refresh_chat_bottom()

    def build_chat_bottom():
        missing_keys = [k for k, v in state["patient_data"].items() if v is None]
        if state["chat_status"]:
            return ft.Row([
                ft.ProgressRing(width=18, height=18, stroke_width=2),
                ft.Text(state["chat_status"], italic=True, opacity=0.7),
            ], spacing=10)

        current_q_key = missing_keys[0] if missing_keys else None
        current_q_label = chat_sym(current_q_key) if current_q_key else None

        text_field = ft.TextField(hint_text=chat_t("chat_input_hint"), expand=True)

        async def on_submit(e):
            await handle_send(e.control.value)

        async def on_click_send(e):
            await handle_send(text_field.value)

        text_field.on_submit = on_submit

        mic_icon = ft.Icons.STOP_CIRCLE_ROUNDED if state["recording"] else ft.Icons.MIC_ROUNDED
        mic_color = RED if state["recording"] else PRIMARY
        mic_tooltip = (chat_t("chat_mic_unavailable") if audio_recorder is None
                       else chat_t("chat_mic_stop") if state["recording"] else chat_t("chat_mic_start"))

        # No patient reply yet: the opening message is a generic warm greeting that hasn't asked
        # about a specific symptom yet, so don't show quick-reply buttons that would mismatch it.
        has_patient_replied = any(m["role"] == "user" for m in state["messages"])

        rows = []
        if missing_keys and has_patient_replied:
            rows.append(ft.Row([
                ft.ElevatedButton(chat_t("chat_quick_yes").format(symptom=current_q_label.lower()),
                                  icon=ft.Icons.CHECK_ROUNDED, bgcolor=PRIMARY, color="white",
                                  on_click=handle_quick_reply(current_q_key, 1, current_q_label)),
                ft.OutlinedButton(chat_t("chat_quick_no"), icon=ft.Icons.CLOSE_ROUNDED,
                                  on_click=handle_quick_reply(current_q_key, 0, current_q_label)),
            ], spacing=10))
        rows.append(ft.Row([text_field,
                            ft.IconButton(icon=mic_icon, icon_color=mic_color, tooltip=mic_tooltip,
                                          on_click=toggle_recording, disabled=audio_recorder is None),
                            ft.IconButton(icon=ft.Icons.SEND_ROUNDED, icon_color=PRIMARY,
                                          on_click=on_click_send)]))
        return ft.Column(rows, spacing=10)

    def refresh_chat_bottom():
        ctrl = state["ui"].get("chat_bottom_ctrl")
        if ctrl is None:
            return
        ctrl.content = build_chat_bottom()
        if ctrl.page:
            ctrl.update()

    def build_chat_view():
        header = ft.Card(elevation=1, content=ft.Container(
            content=ft.Row([
                ft.CircleAvatar(content=ft.Icon(ft.Icons.MEDICAL_SERVICES_ROUNDED, color="white", size=26),
                                 bgcolor=PRIMARY, radius=26),
                ft.Column([
                    ft.Text(t("chat_header_name"), size=20, weight=ft.FontWeight.BOLD, color=PRIMARY),
                    ft.Row([ft.Icon(ft.Icons.CIRCLE, size=10, color=GREEN),
                            ft.Text(t("chat_header_status"), size=13, opacity=0.7)], spacing=6),
                ], spacing=2),
            ], spacing=14),
            padding=15,
        ))

        chat_list = ft.ListView(
            controls=[message_bubble(m) for m in state["messages"]],
            expand=True, spacing=12, auto_scroll=True, padding=10,
        )
        bottom_ctrl = ft.Container(content=build_chat_bottom())

        state["ui"]["chat_list_ctrl"] = chat_list
        state["ui"]["chat_bottom_ctrl"] = bottom_ctrl

        state["ui"]["main_area_ctrl"].content = ft.Column(
            [header, chat_list, bottom_ctrl], expand=True, spacing=15,
        )

    def handle_quick_reply(symptom_key, value, symptom_label):
        def handler(e):
            state["patient_data"][symptom_key] = value
            answer_text = chat_t("quick_reply_yes_echo") if value == 1 else chat_t("quick_reply_no_echo")
            add_chat_bubble("user", f"{answer_text} {symptom_label.lower()}.")

            missing_keys = [k for k, v in state["patient_data"].items() if v is None]
            if len(missing_keys) > 0:
                next_symp = chat_sym(missing_keys[0])
                add_chat_bubble("assistant", chat_t("quick_reply_saved_next").format(symptom=next_symp.lower()))
                refresh_sidebar()
                refresh_chat_bottom()
            else:
                add_chat_bubble("assistant", chat_t("quick_reply_complete"))
                refresh_sidebar()
                navigate_result()
        return handler

    async def detect_language(sample_text):
        normalized = (sample_text or "").strip().lower()
        english_requests = (
            "tiếng anh", "tieng anh", "english", "speak english", "talk in english",
            "reply in english", "answer in english", "use english",
        )
        vietnamese_requests = (
            "tiếng việt", "tieng viet", "vietnamese", "speak vietnamese", "talk in vietnamese",
            "reply in vietnamese", "answer in vietnamese", "use vietnamese",
        )
        if any(phrase in normalized for phrase in english_requests):
            return "en"
        if any(phrase in normalized for phrase in vietnamese_requests):
            return "vi"

        try:
            res = await asyncio.to_thread(
                get_client().chat.completions.create,
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": (
                        "Decide whether the user's message should be answered in Vietnamese or English. "
                        "Reply with ONLY 'vi' or 'en'. If the user asks to switch language, follow the "
                        "requested answer language even if the request itself is written in another language. "
                        "If the message mixes both languages, choose the dominant/current language of the "
                        "user's latest sentence. No other text."
                    )},
                    {"role": "user", "content": sample_text},
                ],
                temperature=0.0, max_tokens=5, extra_body=groq_extra_body(),
            )
            code = res.choices[0].message.content.strip().lower()
            code = "".join(ch for ch in code if ch.isalpha())[:2]
            return code if code in ("vi", "en") else state["ui_language"]
        except Exception:
            return state["ui_language"]

    async def extract_answer(text, current_q_key, current_q_label, other_keys):
        """Dedicated, simple classification call — separated from the 'warm reply' call so a long,
        multi-goal prompt doesn't cause the model to drop the data-extraction task (observed with
        Groq's llama-3.3-70b: bare "có"/"không" answers were sometimes ignored when extraction and
        natural-language generation were asked for in a single call)."""
        other_labels = {k: chat_sym(k) for k in other_keys}
        prompt = f"""
        The patient was just asked (yes/no question) whether they have this symptom: "{current_q_label}".
        The patient's reply: "{text}"

        Classify the reply:
        - "yes" if it confirms the symptom, in ANY language/phrasing, including a single bare word like
          "yes", "yeah", "có", "ừ", "đúng", "uh", "oui".
        - "no" if it denies the symptom, in ANY language/phrasing, including a single bare word like "no",
          "nope", "không", "không bị", "chưa", "non".
        - "unclear" ONLY if the reply truly does not address this yes/no question at all (pure small talk,
          a question back, or a completely different topic).
        A short reply like "không" or "có" alone is ALWAYS a complete, valid answer — never classify a
        clear "có"/"không"/"yes"/"no" as "unclear".

        Also check whether the reply mentions any of these OTHER symptoms (separate from the one above):
        {json.dumps(other_labels, ensure_ascii=False)}
        For each one clearly confirmed or denied, include its exact code with 1 (confirmed) or 0 (denied)
        in "other_symptoms". Only include codes you are confident about.

        Return ONLY this JSON:
        {{"answer": "yes" | "no" | "unclear", "other_symptoms": {{}}, "extra_symptom": "any symptom "
            "mentioned outside all the lists above (e.g. fever, headache) — empty string if none"}}
        """
        res = await asyncio.to_thread(
            get_client().chat.completions.create,
            model=MODEL_NAME, messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.0, extra_body=groq_extra_body(),
        )
        return json.loads(res.choices[0].message.content)

    async def generate_reply(text, lang, next_q_key, next_q_label):
        """Second call: pure natural-language phrasing. Data extraction already happened in
        extract_answer(), so this call only needs to sound warm and ask the (already-correct) next
        question — nothing here can corrupt patient_data anymore."""
        if next_q_key is None:
            task = (
                "The patient's profile is now complete. Write a short, warm closing message telling "
                "them so, and that the system is about to analyze their results. No question needed."
            )
        else:
            task = (
                f'Write a short, warm reply to the patient. If they asked a side question or an unrelated '
                f'health question, answer it briefly as a doctor first, then gently return to the screening. '
                f'Then ask specifically about '
                f'"{next_q_label}" — phrased naturally and flexibly, not a rigid template, not a repeat '
                f'of previous wording. It must end with a clear yes/no question about "{next_q_label}" '
                f"specifically — do not substitute a different symptom."
            )
        prompt = f"""
        You are "AI Doctor LungCare" — a warm, caring respiratory physician who listens, comforts, and
        chats naturally with patients like a real doctor would, not a robotic data-collection form.

        The patient's message: "{text}"

        LANGUAGE RULE: the desired reply language is "{lang}". If "{lang}" is "en", reply in English even
        when the request to switch was written in Vietnamese. If "{lang}" is "vi", reply in Vietnamese even
        when the request to switch was written in English. Otherwise keep common medical terms or short
        bilingual phrases only if the patient naturally mixes them.

        YOUR TASK: {task}

        Return ONLY this JSON: {{"phan_hoi_bac_si": "your full reply here"}}
        """
        res = await asyncio.to_thread(
            get_client().chat.completions.create,
            model=MODEL_NAME, messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.5, extra_body=groq_extra_body(),
        )
        ai_dict = json.loads(res.choices[0].message.content)
        return ai_dict.get("phan_hoi_bac_si", "...")

    async def generate_free_chat_reply(text, lang):
        prompt = f"""
        You are "AI Doctor LungCare" — a warm, caring respiratory physician who listens, comforts, and
        chats naturally with patients like a real doctor would.

        The patient's message: "{text}"

        LANGUAGE RULE: the desired reply language is "{lang}". If "{lang}" is "en", reply in English even
        when the request to switch was written in Vietnamese. If "{lang}" is "vi", reply in Vietnamese even
        when the request to switch was written in English. Otherwise keep common medical terms or short
        bilingual phrases only if the patient naturally mixes them.

        Answer the patient's question helpfully and briefly. You may answer reasonable side questions, but
        keep a medical-safety tone: do not claim to diagnose, recommend urgent care for emergency symptoms,
        and suggest seeing a clinician when symptoms are serious or persistent.

        Return ONLY this JSON: {{"phan_hoi_bac_si": "your full reply here"}}
        """
        res = await asyncio.to_thread(
            get_client().chat.completions.create,
            model=MODEL_NAME, messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.5, extra_body=groq_extra_body(),
        )
        ai_dict = json.loads(res.choices[0].message.content)
        return ai_dict.get("phan_hoi_bac_si", "...")

    async def handle_send(text):
        text = (text or "").strip()
        if not text:
            return
        add_chat_bubble("user", text)
        missing_keys = [k for k, v in state["patient_data"].items() if v is None]
        if not missing_keys:
            state["language"] = await detect_language(text)
            state["chat_status"] = t("chat_thinking")
            refresh_chat_bottom()
            try:
                display_text = await generate_free_chat_reply(text, state["language"])
                add_chat_bubble("assistant", display_text)
            except API_KEY_ERRORS:
                state["api_key_error"] = True
                add_chat_bubble("assistant", t("chat_apikey_error_send"))
            except MODEL_ACCESS_ERRORS:
                add_chat_bubble("assistant", t("chat_model_error").format(model=MODEL_NAME))
            except Exception as ex:
                add_chat_bubble("assistant", t("chat_ai_error").format(err=ex))
            state["chat_status"] = ""
            refresh_chat_bottom()
            return

        state["language"] = await detect_language(text)
        lang = state["language"]

        current_q_key = missing_keys[0]
        current_q_label = chat_sym(current_q_key)
        other_keys = [k for k in missing_keys if k != current_q_key]

        state["chat_status"] = t("chat_thinking")
        refresh_chat_bottom()

        finished = False
        try:
            extract_dict = await extract_answer(text, current_q_key, current_q_label, other_keys)

            answer = extract_dict.get("answer")
            if answer == "yes":
                state["patient_data"][current_q_key] = 1
            elif answer == "no":
                state["patient_data"][current_q_key] = 0

            for k, v in extract_dict.get("other_symptoms", {}).items():
                for act_k in state["patient_data"].keys():
                    if k.strip() == act_k.strip():
                        state["patient_data"][act_k] = v

            extra = extract_dict.get("extra_symptom", "")
            if extra:
                state["extra_symptoms"] += f"[{extra}]. "

            new_missing = [k for k, v in state["patient_data"].items() if v is None]
            next_q_key = new_missing[0] if new_missing else None
            next_q_label = chat_sym(next_q_key) if next_q_key else None

            display_text = await generate_reply(text, lang, next_q_key, next_q_label)
            add_chat_bubble("assistant", display_text)
            if not new_missing:
                finished = True
        except API_KEY_ERRORS:
            state["api_key_error"] = True
            add_chat_bubble("assistant", t("chat_apikey_error_send"))
        except MODEL_ACCESS_ERRORS:
            add_chat_bubble("assistant", t("chat_model_error").format(model=MODEL_NAME))
        except Exception as ex:
            add_chat_bubble("assistant", t("chat_ai_error").format(err=ex))

        state["chat_status"] = ""
        refresh_sidebar()
        if finished:
            navigate_result()
        else:
            refresh_chat_bottom()

    # ---------- RESULT ----------
    def compute_risk():
        if state["risk_score"] is not None:
            return
        d = state["patient_data"]
        if isinstance(ML_MODEL, str):
            state["risk_score"] = "ERROR"
            return
        anx_yel = (1 if d["ANXIETY"] == 1 else 0) * (1 if d["YELLOW_FINGERS"] == 1 else 0)
        input_arr = [[(1 if d[k] == 1 else 0) for k in REQUIRED_FEATURES] + [anx_yel]]
        raw_risk = ML_MODEL.predict_proba(input_arr)[0][1] * 100

        severe_symptoms = d["COUGHING"] + d["CHEST PAIN"] + d["WHEEZING"] + d["SWALLOWING DIFFICULTY"]
        total_symptoms = sum(1 for k, v in d.items() if v == 1)

        if total_symptoms == 0:
            final_risk = 1.5
        elif severe_symptoms == 0 and total_symptoms <= 3:
            final_risk = raw_risk * 0.15
        elif severe_symptoms == 0 and total_symptoms > 3:
            final_risk = raw_risk * 0.35
        elif severe_symptoms == 1:
            final_risk = raw_risk * 0.6
        elif severe_symptoms == 2:
            final_risk = raw_risk * 0.85
        else:
            final_risk = raw_risk

        final_risk = max(1.5, min(final_risk, 98.5))
        state["risk_score"] = round(final_risk, 1)

    async def fetch_advice():
        if state["advice_text"] is not None:
            return
        d = state["patient_data"]
        symptoms_list = [sym(k) for k, v in d.items() if v == 1]
        lang = state["language"] or "vi"
        advice_prompt = (
            f'CRITICAL LANGUAGE RULE: write your ENTIRE reply only in language code "{lang}", every '
            f"sentence, no exceptions.\n\n"
            f"Patient's lung disease risk: {state['risk_score']}%. "
            f"Confirmed symptoms: {', '.join(symptoms_list) if symptoms_list else 'none'}. "
            f"Extra notes: {state['extra_symptoms']}. As a warm, caring respiratory specialist, write one "
            "summary paragraph (may include a reassuring/comforting sentence) and 3 bullet points of "
            f"practical, insightful medical advice. Remember: entire reply in language code \"{lang}\"."
        )
        try:
            res = await asyncio.to_thread(
                get_client().chat.completions.create,
                model=MODEL_NAME, messages=[{"role": "user", "content": advice_prompt}], temperature=0.3,
                extra_body=groq_extra_body(),
            )
            state["advice_text"] = res.choices[0].message.content
        except API_KEY_ERRORS:
            state["api_key_error"] = True
            state["advice_text"] = t("advice_apikey_error")
        except MODEL_ACCESS_ERRORS:
            state["advice_text"] = t("advice_model_error")
        except Exception:
            state["advice_text"] = t("advice_fallback")

        if state["page"] == "result":
            refresh_result_dynamic()
            refresh_sidebar()

    def build_result_static(d, risk):
        if risk >= 70:
            color, icon, title, desc = (
                RED, ft.Icons.DANGEROUS_ROUNDED, t("result_danger_title"), t("result_danger_desc"),
            )
        elif risk >= 35:
            color, icon, title, desc = (
                AMBER, ft.Icons.WARNING_ROUNDED, t("result_warning_title"), t("result_warning_desc"),
            )
        else:
            color, icon, title, desc = (
                GREEN, ft.Icons.CHECK_CIRCLE_ROUNDED, t("result_safe_title"), t("result_safe_desc"),
            )

        alert_card = ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(icon, color=color, size=38),
                        ft.Text(f"{risk}%", size=42, weight=ft.FontWeight.BOLD, color=color)],
                       alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                ft.Text(title, size=16, weight=ft.FontWeight.BOLD, color=color, text_align=ft.TextAlign.CENTER),
                ft.Text(desc, color=color, text_align=ft.TextAlign.CENTER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
            bgcolor=ft.Colors.with_opacity(0.1, color), border=ft.Border.all(2, color),
            border_radius=15, padding=20, shadow=SOFT_SHADOW,
        )

        gauge_color = RED if risk >= 70 else AMBER if risk >= 35 else GREEN
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": gauge_color}, "bgcolor": "rgba(128,128,128,0.2)"},
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(t=30, b=0, l=20, r=20))

        categories = t("radar_categories")
        keys_map = ["COUGHING", "CHEST PAIN", "WHEEZING", "FATIGUE ", "ALLERGY ", "SWALLOWING DIFFICULTY"]
        values = [(1 if d[k] == 1 else 0) for k in keys_map]
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]], theta=categories + [categories[0]], fill="toself",
            fillcolor="rgba(2, 132, 199, 0.4)", line=dict(color=PRIMARY),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 1],
                                         ticktext=[t("sidebar_no"), t("sidebar_yes")]),
                       bgcolor="rgba(128,128,128,0.1)"),
            paper_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(t=30, b=10, l=40, r=40),
        )

        charts_row = ft.ResponsiveRow([
            ft.Container(ft.Card(elevation=2, content=ft.Container(ft.Column([
                ft.Text(t("result_gauge_title"), weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                fc.PlotlyChart(figure=fig_gauge, expand=True),
            ]), padding=15)), col=6),
            ft.Container(ft.Card(elevation=2, content=ft.Container(ft.Column([
                ft.Text(t("result_radar_title"), weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                fc.PlotlyChart(figure=fig_radar, expand=True),
            ]), padding=15)), col=6),
        ])

        return ft.Column([
            ft.Row([ft.Icon(ft.Icons.ASSESSMENT_ROUNDED, color=PRIMARY, size=24),
                    ft.Text(t("result_analysis_title"), size=22, weight=ft.FontWeight.BOLD)], spacing=8),
            alert_card, charts_row,
        ], spacing=15)

    def switch_tab(name):
        def handler(e):
            state["result_tab"] = name
            refresh_result_dynamic()
        return handler

    def build_result_dynamic_content():
        tab_buttons = ft.Row([
            ft.ElevatedButton(
                t("result_tab_advice"), icon=ft.Icons.LIGHTBULB_ROUNDED,
                bgcolor=PRIMARY if state["result_tab"] == "advice" else None,
                color="white" if state["result_tab"] == "advice" else None,
                on_click=switch_tab("advice"),
            ),
            ft.ElevatedButton(
                t("result_tab_hospitals"), icon=ft.Icons.LOCAL_HOSPITAL_ROUNDED,
                bgcolor=PRIMARY if state["result_tab"] == "map" else None,
                color="white" if state["result_tab"] == "map" else None,
                on_click=switch_tab("map"),
            ),
        ])

        if state["result_tab"] == "advice":
            advice_body = (ft.Text(t("result_advice_generating"), italic=True) if state["advice_text"] is None
                           else ft.Markdown(state["advice_text"], selectable=True))
            tab_content = ft.Container(
                content=ft.Column([
                    ft.Row([ft.Icon(ft.Icons.PUSH_PIN_ROUNDED, size=18, color=PRIMARY),
                            ft.Text(t("result_advice_note_title"), size=16, weight=ft.FontWeight.BOLD,
                                    color=PRIMARY)],
                           spacing=6),
                    advice_body,
                ]),
                bgcolor=ft.Colors.with_opacity(0.05, PRIMARY), border=ft.Border.all(1, PRIMARY),
                border_radius=15, padding=20,
            )
        else:
            hospitals = [
                {"name": "Bệnh viện Phổi Trung ương", "lat": 21.0401, "lon": 105.8136,
                 "addr": "463 Hoàng Hoa Thám, Ba Đình, Hà Nội"},
                {"name": "Bệnh viện Phạm Ngọc Thạch", "lat": 10.7558, "lon": 106.6644,
                 "addr": "120 Hùng Vương, Quận 5, TP.HCM"},
                {"name": "Bệnh viện K - Tân Triều", "lat": 20.9632, "lon": 105.7955,
                 "addr": "Số 30 Cầu Bươu, Thanh Trì, Hà Nội"},
            ]
            fig_map = go.Figure(go.Scattermap(
                lat=[h["lat"] for h in hospitals], lon=[h["lon"] for h in hospitals],
                mode="markers", marker=dict(size=14, color=PRIMARY), text=[h["name"] for h in hospitals],
            ))
            fig_map.update_layout(
                map=dict(style="open-street-map", center=dict(lat=16.5, lon=106.5), zoom=4.3),
                margin=dict(t=0, b=0, l=0, r=0), height=300,
            )

            def open_maps(url):
                async def handler(e):
                    await page.launch_url(url)
                return handler

            hospital_cards = []
            for h in hospitals:
                gmap_url = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(h['name'])}"
                hospital_cards.append(ft.Container(
                    content=ft.Column([
                        ft.Text(h["name"], weight=ft.FontWeight.BOLD),
                        ft.Row([ft.Icon(ft.Icons.LOCATION_ON_ROUNDED, size=15, color=GREEN),
                                ft.Text(h["addr"], size=13, expand=True)], spacing=4),
                        ft.TextButton(t("result_map_btn"), icon=ft.Icons.MAP_ROUNDED,
                                      on_click=open_maps(gmap_url)),
                    ], spacing=2),
                    bgcolor=ft.Colors.with_opacity(0.03, PRIMARY),
                    border=ft.Border(left=ft.BorderSide(width=4, color=GREEN)),
                    border_radius=12, padding=15,
                ))

            tab_content = ft.Column([
                ft.Text(t("result_hospitals_intro")),
                fc.PlotlyChart(figure=fig_map, expand=True, height=300),
                *hospital_cards,
            ])

        return ft.Column([tab_buttons, tab_content], spacing=15)

    def refresh_result_dynamic():
        ctrl = state["ui"].get("result_dynamic_ctrl")
        if ctrl is None:
            return
        ctrl.content = build_result_dynamic_content()
        if ctrl.page:
            ctrl.update()

    def build_result_view():
        compute_risk()
        d = state["patient_data"]

        if state["risk_score"] == "ERROR":
            state["ui"]["main_area_ctrl"].content = ft.Column([
                ft.Row([ft.Icon(ft.Icons.DANGEROUS_ROUNDED, color=RED, size=22),
                        ft.Text(t("result_error_title"), color=RED, size=18, weight=ft.FontWeight.BOLD)],
                       spacing=8),
                ft.Row([ft.Icon(ft.Icons.LIGHTBULB_ROUNDED, size=16, opacity=0.8),
                        ft.Text(t("result_error_cause"), opacity=0.8)], spacing=6),
                ft.Text(t("result_error_detail").format(detail=ML_MODEL), size=12, opacity=0.6, selectable=True),
                ft.Text(t("result_error_path").format(path=MODEL_PATH), size=12, opacity=0.6, selectable=True),
            ])
            return

        if "result_static" not in state["ui"]:
            state["ui"]["result_static"] = build_result_static(d, state["risk_score"])

        dynamic_ctrl = ft.Container(content=build_result_dynamic_content())
        state["ui"]["result_dynamic_ctrl"] = dynamic_ctrl

        state["ui"]["main_area_ctrl"].content = ft.Column(
            [state["ui"]["result_static"], dynamic_ctrl],
            spacing=15, scroll=ft.ScrollMode.AUTO, expand=True,
        )
        page.run_task(fetch_advice)

    # ---------- TOP APP BAR (Sáng/Tối + Ngôn ngữ + điều hướng chính) ----------
    def build_appbar():
        return ft.AppBar(
            leading=ft.Container(
                content=ft.Icon(ft.Icons.HEALTH_AND_SAFETY_ROUNDED, color="white", size=26),
                padding=ft.Padding.only(left=14),
            ),
            leading_width=48,
            title=ft.Text("AI LungCare", weight=ft.FontWeight.BOLD, size=20, color="white"),
            center_title=False,
            bgcolor=PRIMARY,
            actions=[
                ft.IconButton(icon=ft.Icons.HOME_ROUNDED, icon_color="white", tooltip=t("tooltip_home"),
                              on_click=navigate_home),
                ft.IconButton(icon=ft.Icons.IMAGE_SEARCH_ROUNDED, icon_color="white", tooltip=t("tooltip_image"),
                              on_click=navigate_image),
                ft.TextButton(
                    content=ft.Text("EN" if state["ui_language"] == "vi" else "VI", color="white",
                                     weight=ft.FontWeight.BOLD),
                    tooltip=t("tooltip_lang"), on_click=toggle_ui_language,
                ),
                ft.IconButton(
                    icon=ft.Icons.LIGHT_MODE_ROUNDED if state["dark_mode"] else ft.Icons.DARK_MODE_ROUNDED,
                    icon_color="white",
                    tooltip=t("tooltip_dark_off") if state["dark_mode"] else t("tooltip_dark_on"),
                    on_click=toggle_theme,
                ),
                ft.Container(width=8),
            ],
        )

    def toggle_theme(e):
        state["dark_mode"] = not state["dark_mode"]
        page.theme_mode = ft.ThemeMode.DARK if state["dark_mode"] else ft.ThemeMode.LIGHT
        page.appbar = build_appbar()
        page.update()

    def toggle_ui_language(e):
        state["ui_language"] = "en" if state["ui_language"] == "vi" else "vi"
        page.title = t("window_title")
        page.appbar = build_appbar()
        refresh_current_page()
        page.update()

    page.title = t("window_title")
    page.appbar = build_appbar()
    page.update()

    if not state["groq_api_key"]:
        navigate_gate()
    else:
        navigate_home()


if __name__ == "__main__":
    flet_view = os.environ.get("LUNGCARE_FLET_VIEW", "desktop").strip().lower()
    flet_host = os.environ.get("LUNGCARE_HOST", "127.0.0.1")
    flet_port = int(os.environ.get("LUNGCARE_PORT", "8550"))

    if flet_view in ("web", "browser", "web_browser"):
        ft.run(main, view=ft.AppView.WEB_BROWSER, host=flet_host, port=flet_port)
    elif flet_view in ("server", "web_server"):
        ft.run(main, view=ft.AppView.FLET_APP_WEB, host=flet_host, port=flet_port)
    else:
        ft.run(main)
