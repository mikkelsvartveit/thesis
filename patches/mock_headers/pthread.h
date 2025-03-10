#ifndef MOCK_PTHREAD_H
#define MOCK_PTHREAD_H

#ifdef __cplusplus
extern "C" {
#endif

/* Basic pthread types */
typedef unsigned long int pthread_t;
typedef struct {
    int dummy;
} pthread_attr_t;

typedef struct {
    int dummy;
} pthread_mutex_t;

typedef struct {
    int dummy;
} pthread_mutexattr_t;

typedef struct {
    int dummy;
} pthread_cond_t;

typedef struct {
    int dummy;
} pthread_condattr_t;

typedef struct {
    int dummy;
} pthread_rwlock_t;

typedef struct {
    int dummy;
} pthread_rwlockattr_t;

typedef struct {
    int dummy;
} pthread_spinlock_t;

typedef struct {
    int dummy;
} pthread_barrier_t;

typedef struct {
    int dummy;
} pthread_barrierattr_t;

typedef struct {
    int dummy;
} pthread_key_t;

typedef struct {
    int dummy;
} pthread_once_t;

/* Define common pthread constants */
#define PTHREAD_MUTEX_INITIALIZER {0}
#define PTHREAD_RWLOCK_INITIALIZER {0}
#define PTHREAD_COND_INITIALIZER {0}
#define PTHREAD_ONCE_INIT {0}

/* Mutex types */
enum {
    PTHREAD_MUTEX_NORMAL,
    PTHREAD_MUTEX_RECURSIVE,
    PTHREAD_MUTEX_ERRORCHECK,
    PTHREAD_MUTEX_DEFAULT = PTHREAD_MUTEX_NORMAL
};

/* Function implementations - all return 0 for success */

/* Thread management */
static inline int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void*), void *arg) {
    return 0;
}

static inline int pthread_join(pthread_t thread, void **retval) {
    return 0;
}

static inline int pthread_detach(pthread_t thread) {
    return 0;
}

static inline pthread_t pthread_self(void) {
    return 0;
}

static inline int pthread_equal(pthread_t t1, pthread_t t2) {
    return 1;
}

static inline int pthread_attr_init(pthread_attr_t *attr) {
    return 0;
}

static inline int pthread_attr_destroy(pthread_attr_t *attr) {
    return 0;
}

/* Mutex functions */
static inline int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr) {
    return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t *mutex) {
    return 0;
}

static inline int pthread_mutex_lock(pthread_mutex_t *mutex) {
    return 0;
}

static inline int pthread_mutex_trylock(pthread_mutex_t *mutex) {
    return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *mutex) {
    return 0;
}

static inline int pthread_mutexattr_init(pthread_mutexattr_t *attr) {
    return 0;
}

static inline int pthread_mutexattr_destroy(pthread_mutexattr_t *attr) {
    return 0;
}

static inline int pthread_mutexattr_settype(pthread_mutexattr_t *attr, int type) {
    return 0;
}

/* Condition variable functions */
static inline int pthread_cond_init(pthread_cond_t *cond, const pthread_condattr_t *attr) {
    return 0;
}

static inline int pthread_cond_destroy(pthread_cond_t *cond) {
    return 0;
}

static inline int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
    return 0;
}

static inline int pthread_cond_timedwait(pthread_cond_t *cond, pthread_mutex_t *mutex, const struct timespec *abstime) {
    return 0;
}

static inline int pthread_cond_signal(pthread_cond_t *cond) {
    return 0;
}

static inline int pthread_cond_broadcast(pthread_cond_t *cond) {
    return 0;
}

/* RW lock functions */
static inline int pthread_rwlock_init(pthread_rwlock_t *rwlock, const pthread_rwlockattr_t *attr) {
    return 0;
}

static inline int pthread_rwlock_destroy(pthread_rwlock_t *rwlock) {
    return 0;
}

static inline int pthread_rwlock_rdlock(pthread_rwlock_t *rwlock) {
    return 0;
}

static inline int pthread_rwlock_tryrdlock(pthread_rwlock_t *rwlock) {
    return 0;
}

static inline int pthread_rwlock_wrlock(pthread_rwlock_t *rwlock) {
    return 0;
}

static inline int pthread_rwlock_trywrlock(pthread_rwlock_t *rwlock) {
    return 0;
}

static inline int pthread_rwlock_unlock(pthread_rwlock_t *rwlock) {
    return 0;
}

/* Thread-specific data */
static inline int pthread_key_create(pthread_key_t *key, void (*destructor)(void*)) {
    return 0;
}

static inline int pthread_key_delete(pthread_key_t key) {
    return 0;
}

static inline int pthread_setspecific(pthread_key_t key, const void *value) {
    return 0;
}

static inline void* pthread_getspecific(pthread_key_t key) {
    return NULL;
}

/* Once-only initialization */
static inline int pthread_once(pthread_once_t *once_control, void (*init_routine)(void)) {
    return 0;
}

/* Thread cancelation (stubbed) */
static inline int pthread_cancel(pthread_t thread) {
    return 0;
}

/* Yield function */
static inline int pthread_yield(void) {
    return 0;
}

static inline int sched_yield(void) {
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* MOCK_PTHREAD_H */