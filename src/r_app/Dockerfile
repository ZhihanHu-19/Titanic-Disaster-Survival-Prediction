FROM rocker/r-base:4.3.1

WORKDIR /app

COPY src/r_app/install_packages.R .
COPY src/r_app/r_app.R .

RUN Rscript install_packages.R

CMD ["Rscript", "r_app.R"]