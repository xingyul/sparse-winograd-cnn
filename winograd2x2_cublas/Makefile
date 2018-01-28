
SUBDIRS = $(shell ls -d */)
all:
	for dir in $(SUBDIRS) ; do \
        make -C  $$dir ; \
    done

clean:
	for dir in $(SUBDIRS) ; do \
        make --directory=$$dir clean; \
    done
