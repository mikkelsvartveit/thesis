#ifndef POLL_H
#define POLL_H

#include <time.h>
#include <sys/types.h>

/* Basic poll definitions */
#define POLLIN      0x0001
#define POLLPRI     0x0002
#define POLLOUT     0x0004
#define POLLERR     0x0008
#define POLLHUP     0x0010
#define POLLNVAL    0x0020

struct pollfd {
    int fd;           /* File descriptor */
    short events;     /* Requested events */
    short revents;    /* Returned events */
};

/* Stubbed implementation that will return immediately */
static inline int poll(struct pollfd *fds, unsigned long nfds, int timeout) {
    /* Set POLLIN for all file descriptors to simulate readiness */
    for (unsigned long i = 0; i < nfds; i++) {
        fds[i].revents = POLLIN;
    }
    return nfds; /* Return number of file descriptors as if all are ready */
}

#endif /* POLL_H */