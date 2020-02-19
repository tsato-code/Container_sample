from invoke import task


@task
def stop(ctx):
    cmd = "docker stop {}".format(ctx.container_name)
    print(cmd)
    ctx.run(cmd)


@task(stop)
def remove(ctx):
    cmd = "docker rm {}".format(ctx.container_name)
    print(cmd)
    ctx.run(cmd)


@task
def run(ctx):
    cmd = "docker run -t -d --name {} -v $PWD:/tmp/working gcr.io/kaggle-images/python:v73".format(ctx.container_name)
    print(cmd)
    ctx.run(cmd)


@task
def prepare(ctx):
    cmd = "docker exec -i -w /tmp/working {} python project/source/prepare.py".format(ctx.container_name)
    print(cmd)
    ctx.run(cmd)


@task
def train(ctx):
    cmd = "docker exec -i -w /tmp/working {} python project/source/train.py".format(ctx.container_name)
    print(cmd)
    ctx.run(cmd)


@task
def evaluate(ctx):
    cmd = "docker exec -i -w /tmp/working {} python project/source/evaluate.py".format(ctx.container_name)
    ctx.run(cmd)

