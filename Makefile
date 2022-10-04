# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ncolin <ncolin@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/10/04 15:45:11 by ncolin            #+#    #+#              #
#    Updated: 2022/10/04 15:45:27 by ncolin           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

all:
	@echo Setting up virtual env
	python3 -m pip install --user virtualenv
	python3 -m venv .venv
	( \
       source .venv/bin/activate; \
       python3 -m pip install -r ./requirements.txt; \
    )
	

clean:
	( \
       rm -rf .venv thetas.json; \
    )
