FROM node:22-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY --chown=node:node *.js ./
COPY --chown=node:node *.sh ./
COPY --chown=node:node assets ./assets
RUN chmod 755 ./*.sh

USER node

EXPOSE 8080
CMD ["node", "server.js"]
