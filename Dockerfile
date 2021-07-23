# Add a line here to specify the docker image to inherit from.
# FROM registry.aisingapore.net/aiap/polyaxon/pytorch-tf2-cpu:latest
FROM continuumio/miniconda3:4.9.2

ARG WORK_DIR="/assignment8"

WORKDIR $WORK_DIR
RUN mkdir -p $WORK_DIR && chown -R 2222:2222 $WORK_DIR && cd $WORK_DIR

# Add lines here to copy over your src folder and 
# any other files you need in the image (like the saved model).
# Offloaded to `.dockerignore` file, lift-&-shift `COPY` entire context with ignores
RUN bash -c "mkdir -p ./{model,src}"
COPY src src
COPY model model
COPY conda.yml ./

# Add a line here to update the base conda environment using the conda.yml. 
RUN pip install --upgrade pip
RUN conda update -n base -c defaults conda && \
    conda env update -f conda.yml -n base &&\
    echo "source activate base" >> ~/.bashrc && \
    ln -sf $(which python3) /usr/local/bin/python
#RUN conda update -n base -c defaults conda && \
#    conda env update -f conda.yml -n base &&\
#    echo "source activate base" >> ~/.bashrc

# DO NOT remove the following line - it is required for deployment on Tekong
RUN chown -R 1000450000:0 $WORK_DIR
# `chown` above would have transferred all permissions to OpenShift at group 0

EXPOSE 8000

CMD ["ls"]

# Add a line here to run your app
CMD [ "python", "-m" , "src.app", "run", "--host=0.0.0.0"]