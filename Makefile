.PHONY: deploy restart status verify

deploy:
	git pull origin main
	sudo systemctl restart orion-rag.service
	sleep 2
	systemctl status orion-rag.service --no-pager
	curl -I http://127.0.0.1:5000/api/health

restart:
	sudo systemctl restart orion-rag.service

status:
	systemctl status orion-rag.service --no-pager

verify:
	curl -i http://127.0.0.1:5000/api/health
