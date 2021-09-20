FROM python:3.7

COPY . /streamlit_app 
WORKDIR /streamlit_app
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]

CMD ["prediction.py"]
